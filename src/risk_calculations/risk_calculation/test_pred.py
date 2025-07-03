"""
Author: Dominika Malinowska

This script generates infrastructure map from OSM data and uses a pre-trained
machine learning model to predict PS density.
"""

# Set-up

# Import packages

import os
import time
from typing import Any, Optional, Union, cast
import numpy as np
import osmium
import shapely
import shapely.wkb as wkblib
import matplotlib.pyplot as plt
import timm
from timm.data.constants import WeightsEnum
from torch.optim.lr_scheduler import ReduceLROnPlateau
from kornia import augmentation as K

from shapely.geometry import Polygon
from torchgeo.datasets import (
    GeoDataset,
    RasterDataset,
    stack_samples,
    unbind_samples,
)
from torchgeo.datasets.splits import random_bbox_splitting
from torchgeo.datamodules import GeoDataModule
from torchgeo.samplers import (
    GridGeoSampler,
    RandomGeoSampler,
)
from torchgeo.models import FCN, get_weight
from torchgeo.trainers import utils
from torchgeo.transforms import AugmentationSequential

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
)
from lightning.pytorch import LightningModule
import segmentation_models_pytorch as smp
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from rasterio.merge import merge

# import cProfile
# import pstats
# import io

# Set environment variable
os.environ["USE_PYGEOS"] = "0"


# Function definitions #
def create_tile_poly_files(data_path, min_lat, max_lat, min_lon, max_lon):
    """
    This function creates polygon files for tiles based on the provided latitude and longitude ranges.

    Parameters:
    data_path (str): The path where the polygon files will be saved.
    min_lat (float): The minimum latitude.
    max_lat (float): The maximum latitude.
    min_lon (float): The minimum longitude.
    max_lon (float): The maximum longitude.

    Returns:
    str: The path of the polygon file.
    """

    # Create output directory for .poly files
    poly_dir = os.path.join(data_path, "tile_poly_files")

    # If the directory does not exist, create it
    if not os.path.exists(poly_dir):
        os.makedirs(poly_dir)

    # Generate tiles for the given latitude and longitude ranges
    latitudes = np.arange(min_lat, max_lat, 1)
    longitudes = np.arange(min_lon, max_lon, 1)

    # Iterate over all latitude and longitude pairs
    for lat in latitudes:
        for lon in longitudes:
            # Create a name for the tile based on coordinates
            tile_name = f'{"S" if lat < 0 else "N"}{abs(lat)}{"E" if lon > 0 else "W"}{abs(lon)}'

            # Create a square polygon representing the tile
            polygon = Polygon(
                [
                    (lon, lat),
                    (lon + 1, lat),
                    (lon + 1, lat + 1),
                    (lon, lat + 1),
                    (lon, lat),
                ]
            )

            # Write the .poly file
            poly_file_path = os.path.join(poly_dir, f"{tile_name}.poly")
            with open(poly_file_path, "w") as f:
                f.write(tile_name + "\n")
                f.write("0\n")  # Assuming only one exterior ring
                for coord in polygon.exterior.coords:
                    f.write(f"    {coord[0]}     {coord[1]}\n")
                f.write("END\n")

            return poly_file_path


def clip_osm_osmconvert(input_osm_path, area_poly, output_osm_path, osm_convert_path):
    # Adapted from: https://github.com/ElcoK/osm_clipper/blob/master/src/osm_clipper.py
    """
    Clip the input o5m file covering a continent or planet using the poly file as boundary and
    save to a new o5m file. This function uses the osmconvert tool, which can be found at
    http://wiki.openstreetmap.org/wiki/Osmconvert.

    Arguments:
        input_osm_path (str): Path to the osm.pbf file to be clipped.
        area_poly (str): Path to the .poly file, made through the 'create_poly_files' function.
        output_osm_path (str): Path indicating the final output dir and output name of the new .o5m file.
        osm_convert_path (str): Path to the osmconvert tool.

    Returns:
        None. Outputs a clipped .o5m file.

    Note: Other file extensions such as osm and osm.pbf should also work fine.
    Note2: Dropping relations makes the processing faster and generates a smaller file.
        However, it also removes all items that are partially outside of the border.
        Therefore, to retain all information no relations will be dropped.
    """
    print("{} started!".format(output_osm_path), flush=True)

    # Check if the output file already exists
    if not os.path.exists(output_osm_path):
        # Construct the command to clip the OSM file
        command = (
            f"{osm_convert_path} {input_osm_path} -B={area_poly} "
            "--complete-ways --complete-multipolygons "
            f"-o={output_osm_path}"
        )

        try:
            # Execute the command
            os.system(command)
            print(f"{output_osm_path} finished!")
        except Exception as e:
            print(f"{output_osm_path} did not finish! Error: {e}")


def convert_osm_osmconvert(input_osm_path, output_osm_path):
    """
    TBD
    """
    print("{} started!".format(output_osm_path), flush=True)

    # Check if the output file already exists
    if not os.path.exists(output_osm_path):
        # Construct the command to convert the OSM file

        command = f"osmconvert64 {input_osm_path} -o={output_osm_path}"

        try:
            # Execute the command
            os.system(command)
            print(f"{output_osm_path} finished!")
        except Exception as e:
            print(f"{output_osm_path} did not finish! Error: {e}")


class BuildRoadRailHandler(osmium.SimpleHandler):
    # credit: https://max-coding.medium.com/getting-administrative-boundaries-
    # from-open-street-map-osm-using-pyosmium-9f108c34f86
    # https://medium.com/@max-coding/extracting-open-street-map-osm-street-data-
    # from-data-files-using-pyosmium-afca6eaa5d00
    """
    This class is used to handle OSM data. It extends the SimpleHandler class from the osmium package.
    It extracts buildings, roads, and railways from the OSM data.
    """

    def __init__(self):
        # Call the parent class's constructor
        osmium.SimpleHandler.__init__(self)

        # Initialize lists to store buildings, roads, and railways
        self.buildings = []
        self.roads = []
        self.railways = []

        # Initialize a WKBFactory, which will be used to create geometries
        self.wkbfab = osmium.geom.WKBFactory()

    def area(self, a):
        """
        This method is called for each area in the OSM data.
        If the area is a building, it is added to the buildings list.
        """
        if "building" in a.tags:
            try:
                # Create a multipolygon geometry for the area
                wkbshape = self.wkbfab.create_multipolygon(a)
                # Convert the WKB shape to a Shapely object
                shapely_obj = shapely.wkb.loads(wkbshape, hex=True)

                # Create a dictionary representing the building
                building = {"id": a.id, "geometry": shapely_obj}

                # Add the building to the buildings list
                self.buildings.append(building)
            except RuntimeError as e:
                print(f"An error occurred while processing area with id {a.id}: {e}")

    def way(self, w):
        """
        This method is called for each way in the OSM data.
        If the way is a road or a railway, it is added to the corresponding list.
        """
        if w.tags.get("highway") in ["motorway", "trunk", "primary", "secondary"]:
            try:
                # Create a linestring geometry for the way
                wkb = self.wkbfab.create_linestring(w)
                # Convert the WKB shape to a Shapely object
                geo = wkblib.loads(wkb, hex=True)

            except Exception:
                # If an error occurs while creating the geometry, skip this way
                return

            # Create a dictionary representing the road
            row = {"id": w.id, "geometry": geo}

            # Add highway and bridge columns
            for key, value in w.tags:
                if key in ["highway", "bridge"]:
                    row[key] = value

            # Add the road to the roads list
            self.roads.append(row)

        elif w.tags.get("railway") == "rail":
            try:
                # Create a linestring geometry for the way
                wkb = self.wkbfab.create_linestring(w)
                # Convert the WKB shape to a Shapely object
                geo = wkblib.loads(wkb, hex=True)
                # Create a dictionary representing the railway
                row = {"id": w.id, "geometry": geo, "bridge": w.tags["bridge"]}

            except KeyError:
                # except when there is no info about bridge (it'll set bridge to NaN)
                row = {"id": w.id, "geometry": geo}

            except Exception:
                # If an error occurs while creating the geometry, skip this way
                return

            # Add the railway to the railways list
            self.railways.append(row)


class RegressionTask(LightningModule):
    """
    LightningModule for training models on regression datasets.
    This class is a copy from TorchGeo with some modifications (e.g. plot functions).
    """

    target_key: str = "label"

    def plot(self, sample):
        """
        Plot the image, mask, and prediction.
        """
        image1 = sample["image"][0]
        mask = sample["mask"]
        prediction = sample["prediction"]

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(3 * 5, 5))
        axs[0].imshow(image1.squeeze().numpy(), cmap="turbo", vmin=0, vmax=1)
        axs[0].axis("off")
        axs[1].imshow(mask, vmin=0, vmax=80)
        axs[1].axis("off")
        axs[2].imshow(prediction.detach().numpy(), vmin=0, vmax=80)
        axs[2].axis("off")

        axs[0].set_title("Image")
        axs[1].set_title("Mask")
        axs[2].set_title("Prediction")

        plt.tight_layout()

        return fig

    def config_model(self) -> None:
        """
        Configures the model based on hyperparameters.
        """
        # Create model
        weights = self.hyperparams["weights"]
        self.model = timm.create_model(
            self.hyperparams["model"],
            num_classes=self.hyperparams["num_classes"],
            in_chans=self.hyperparams["in_channels"],
            pretrained=weights is True,
        )

        # Load weights
        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            elif os.path.exists(weights):
                _, state_dict = utils.extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            self.model = utils.load_state_dict(self.model, state_dict)

        # Freeze backbone and unfreeze classifier head
        if self.hyperparams.get("freeze_backbone", False):
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True

    def config_task(self) -> None:
        """
        Configures the task based on kwargs parameters.
        """
        self.config_model()

        self.loss: nn.Module
        if self.hyperparams["loss"] == "mse":
            self.loss = nn.MSELoss()
        elif self.hyperparams["loss"] == "mae":
            self.loss = nn.L1Loss()
        elif self.hyperparams["loss"] == "huber":
            self.loss = nn.HuberLoss(reduction="mean", delta=1.0)
        else:
            raise ValueError(
                f"Loss type '{self.hyperparams['loss']}' is not valid. "
                f"Currently, supports 'mse' or 'mae' loss."
            )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LightningModule for training simple regression models.

        Keyword Args:
            model: Name of the timm model to use
            weights: Either a weight enum, the string representation of a weight enum,
                True for ImageNet weights, False or None for random weights,
                or the path to a saved model state dict.
            num_outputs: Number of prediction outputs
            in_channels: Number of input channels to model
            learning_rate: Learning rate for optimizer
            lr_patience: Patience for learning rate scheduler
            freeze_backbone: Freeze the backbone network to linear probe
                the regression head. Does not support FCN models.
            freeze_decoder: Freeze the decoder network to linear probe
                the regression head. Does not support FCN models.
                Only applicable to PixelwiseRegressionTask.

        .. versionchanged:: 0.4
            Change regression model support from torchvision.models to timm

        .. versionadded:: 0.5
           The *freeze_backbone* and *freeze_decoder* parameters.
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()
        self.hyperparams = cast(dict[str, Any], self.hparams)
        self.config_task()

        # Initialize metrics
        self.train_metrics = MetricCollection(
            {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        self.prediction_metrics = self.train_metrics.clone(prefix="prediction_")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        batch = args[0]
        # batch_idx = args[1]

        # Data augumentation
        # see https://github.com/pytorch/vision/issues/566
        # and https://github.com/microsoft/torchgeo/discussions/1417
        transforms = AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation([90, 90], p=0.5),
            data_keys=["image", "mask"],
        )
        batch = transforms(batch)

        x = batch["image"]
        y = batch[self.target_key]
        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        loss: Tensor = self.loss(y_hat, y.to(torch.float))
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True
        )  # logging to TensorBoard
        self.train_metrics(y_hat, y.to(torch.float))

        #         y_hat_hard = y_hat.squeeze(dim=1)

        #         if batch_idx < 10:
        #             batch["prediction"] = y_hat_hard
        #             for key in ["image", "mask", "prediction"]:
        #                 batch[key] = batch[key].cpu()
        #             sample = unbind_samples(batch)[0]
        #             fig = self.plot(sample)
        #             summary_writer = self.logger.experiment
        #             summary_writer.add_figure(
        #                 f"image/train/{batch_idx}", fig, global_step=self.global_step
        #             )
        #             plt.close()

        return loss

    def on_train_epoch_end(self) -> None:
        """Logs epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
        self.train_metrics.reset()

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        y = batch[self.target_key]
        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        y_hat_hard = y_hat.squeeze(dim=1)

        if batch_idx < 10:
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]
            fig = self.plot(sample)
            summary_writer = self.logger.experiment
            summary_writer.add_figure(
                f"image/val/{batch_idx}", fig, global_step=self.global_step
            )

            plt.close()

        loss: Tensor = self.loss(y_hat, y.to(torch.float))
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.val_metrics(y_hat, y.to(torch.float))

        return loss

    def on_validation_epoch_end(self) -> None:
        """Logs epoch level validation metrics."""
        self.log_dict(
            self.val_metrics.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute test loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        y = batch[self.target_key]
        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        loss = self.loss(y_hat, y.to(torch.float))
        self.log("test_loss", loss)
        self.test_metrics(y_hat, y.to(torch.float))

        y_hat_hard = y_hat.squeeze(dim=1)

        if batch_idx < 10:
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]
            fig = self.plot(sample)
            summary_writer = self.logger.experiment
            summary_writer.add_figure(
                f"image/test/{batch_idx}", fig, global_step=self.global_step
            )
            plt.close()

        return loss

    def on_test_epoch_end(self) -> None:
        """Logs epoch level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the predictions.

        Args:
            batch: the output of your DataLoader
        Returns:
            predicted values
        """
        #         batch = args[0]
        #         x = batch["image"]
        #         y_hat: Tensor = self(x)

        batch = args[0]
        # batch_idx = args[1]
        x = batch["image"]
        y = batch[self.target_key]
        y_hat: Tensor = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        mae_calc = nn.L1Loss()
        mae = mae_calc(y_hat, y.to(torch.float))
        self.prediction_metrics(y_hat, y.to(torch.float))
        return {"y_pred": y_hat, "y_true": y, "MAE": mae}

    def prediction_epoch_end(outputs):
        y_preds = torch.cat([output["y_pred"] for output in outputs])
        y_trues = torch.cat([output["y_true"] for output in outputs]).to(torch.float)

        mae_calc = nn.L1Loss()
        mae = mae_calc(y_preds, y_trues)

        # Log the metrics
        print(f"Mean Absolute Error: {mae.item()}")

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            learning rate dictionary
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hyperparams["learning_rate"],
            betas=(self.hyperparams["beta1"], self.hyperparams["beta2"]),
            weight_decay=self.hyperparams["wd"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hyperparams["lr_patience"],
                    threshold=self.hyperparams["min_delta"],
                ),
                "monitor": "val_loss",
            },
        }


class PixelwiseRegressionTask(RegressionTask):
    # Class copied from the TorchGeo so that some alternations can be made
    """LightningModule for pixelwise regression of images.

    Supports `Segmentation Models Pytorch
    <https://github.com/qubvel/segmentation_models.pytorch>`_
    as an architecture choice in combination with any of these
    `TIMM backbones <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_.

    .. versionadded:: 0.5
    """

    target_key: str = "mask"

    def config_model(self) -> None:
        """Configures the model based on kwargs parameters."""
        weights = self.hyperparams["weights"]

        if self.hyperparams["model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hyperparams["backbone"],
                encoder_weights="imagenet" if weights is True else None,
                in_channels=self.hyperparams["in_channels"],
                classes=1,
                activation="identity",
                #                 decoder_attention_type='scse'
            )
        elif self.hyperparams["model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hyperparams["backbone"],
                encoder_weights="imagenet" if weights is True else None,
                in_channels=self.hyperparams["in_channels"],
                classes=1,
            )
        elif self.hyperparams["model"] == "fcn":
            self.model = FCN(
                in_channels=self.hyperparams["in_channels"],
                classes=1,
                num_filters=self.hyperparams["num_filters"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hyperparams['model']}' is not valid. "
                f"Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if self.hyperparams["model"] != "fcn":
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hyperparams.get("freeze_backbone", False) and self.hyperparams[
            "model"
        ] in ["unet", "deeplabv3+"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hyperparams.get("freeze_decoder", False) and self.hyperparams[
            "model"
        ] in ["unet", "deeplabv3+"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False


class CustomGeoDataModule(GeoDataModule):
    # define data module for handling the dataset
    def __init__(
        self,
        dataset_class: type[GeoDataset],
        dataset_in: Dataset[dict[str, Tensor]],
        batch_size: int = 1,
        patch_size: Union[int, tuple[int, int]] = 64,
        length: Optional[int] = None,
        length_tr: Optional[int] = None,
        length_val: Optional[int] = None,
        length_pred: Optional[int] = None,
        num_workers: int = 0,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        # BaseDataModule is normalizing images in all batches with mean = 0 and std =255.
        # That doesn't make sense when images aren't RGB (i.e. in range 0-255)
        # Thus, change std to 1 to avoid the normalization
        self.std = torch.tensor(1)

        # Define input dataset
        self.dataset_in = dataset_in
        self.seed = seed
        self.generator = torch.Generator().manual_seed(self.seed)

        def on_after_batch_transfer(
            self, batch: dict[str, Tensor], dataloader_idx: int
        ) -> dict[str, Tensor]:
            """Apply batch augmentations to the batch after it is transferred to the device.

            Args:
                batch: A batch of data that needs to be altered or augmented.
                dataloader_idx: The index of the dataloader to which the batch belongs.

            Returns:
                A batch of data.
            """
            if self.trainer:
                if self.trainer.training:
                    split = "train"
                elif self.trainer.validating or self.trainer.sanity_checking:
                    split = "val"
                elif self.trainer.testing:
                    split = "test"
                elif self.trainer.predicting:
                    split = "predict"

                aug = self._valid_attribute(f"{split}_aug", "aug")
                batch = aug(batch)

            return batch

        super().__init__(
            dataset_class, batch_size, patch_size, length, num_workers, **kwargs
        )
        self.length_tr = length_tr
        self.length_val = length_val
        self.length_pred = length_pred

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        #         self.dataset = self.dataset_class(**self.kwargs)
        self.dataset = self.dataset_in

        # random_bbox_splitting: A 0.8/0.2 split will put 80% of each file in one dataset
        # and the other 20% in the other dataset. Every file will occur in both datasets.
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = random_bbox_splitting(
            self.dataset,
            [0.7, 0.15, 0.15],
            generator=self.generator,
        )

        self.predict_dataset = self.dataset

        # Define samplers used for different sateges. For trainging, validation, and test use random geo sampler,
        # so that the patches are randomly selected from the full spatial extend.
        if stage in ["fit"]:
            self.train_batch_sampler = RandomGeoSampler(
                self.train_dataset, self.patch_size, self.length_tr
            )

        if stage in ["fit", "validate"]:
            #             self.val_sampler = RandomGeoSampler(
            #                 self.val_dataset, self.patch_size, self.length_val
            #             )

            self.val_sampler = GridGeoSampler(
                # Usually the stride should be slightly smaller than the chip size such that each chip
                # has some small overlap with surrounding chips.
                self.val_dataset,
                self.patch_size,
                self.patch_size - 8,
            )

        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                # Usually the stride should be slightly smaller than the chip size such that each chip
                # has some small overlap with surrounding chips.
                self.test_dataset,
                self.patch_size,
                self.patch_size - 8,
            )
        # For prediction, use grid geo sampler?
        if stage in ["predict"]:
            self.predict_sampler = GridGeoSampler(
                self.predict_dataset,
                self.patch_size,
                self.patch_size,
            )

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            generator=self.generator,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        #         print('val_dataloader')

        #         val_dataloader =DataLoader(
        #             dataset=self.val_dataset,
        #             batch_size=self.batch_size,
        #             sampler=self.val_sampler,
        #             num_workers=self.num_workers,
        #             collate_fn=self.collate_fn,
        #         )

        #         for b, batch in enumerate(val_dataloader):
        #             if b<10:
        #                 print(batch['bbox'])

        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            generator=self.generator,
        )

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            generator=self.generator,
        )

    def predict_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.batch_size,
            sampler=self.predict_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            generator=self.generator,
            # drop_last = True
        )


class RasterDataset_imgs_coh(RasterDataset):
    """
    This class represents a raster dataset with coherence images.
    It extends the RasterDataset class and sets the filename_glob and is_image attributes.
    """

    # The filename pattern to match for coherence images
    filename_glob = "*rho.tif"
    is_image = True


class RasterDataset_imgs_infmap(RasterDataset):
    """
    This class represents a raster dataset with infrastructure map images.
    It extends the RasterDataset class and sets the filename_glob and is_image attributes.
    """

    # The filename pattern to match for infrastructure map images
    filename_glob = "*.tif"
    is_image = True


def scale_coh(item: dict):
    """
    Scale the coherence data in the given item and convert it from int to float.

    Parameters:
    item (dict): A dictionary containing the coherence data. The data is assumed to be in the "image" key.

    Returns:
    dict: The same dictionary, but with the coherence data scaled and converted to float.
    """
    # Scale the coherence data by dividing by 1000.0
    item["image"] = item["image"] / 1000.0

    # Return the modified item
    return item


def create_in_memory_geochip(predicted_chip, geotransform, crs):
    # Source:
    # https://www.kaggle.com/code/luizclaudioandrade/torchgeo-101#Making-predictions,-applying-the-correct-georrefence-to-them-and-merging-them-in-one-file-that-is-saved-to-disk.

    """
    Apply georeferencing to the predicted chip.

    Parameters:
        predicted_chip (numpy array): The predicted segmentation chip (e.g., binary mask).
        geotransform (tuple): A tuple containing the geotransformation
        information of the chip (x-coordinate of the top left corner, x and y pixel size,
        rotation, y-coordinate of the top left corner, and rotation).
        crs (str): Spatial Reference System (e.g., EPSG code) of the chip.

    Return:
        A rasterio dataset that is georeferenced.
    """
    # Create a new in-memory file
    memfile = MemoryFile()

    # Open the in-memory file as a dataset
    dataset = memfile.open(
        driver="GTiff",  # We are creating a GeoTIFF
        height=predicted_chip.shape[2],  # The height of the chip
        width=predicted_chip.shape[3],  # The width of the chip
        count=predicted_chip.shape[1],  # The number of bands in the chip
        dtype=np.uint8,  # The data type of the chip
        crs=crs,  # The coordinate reference system of the chip
        transform=geotransform,  # The geotransformation to apply to the chip
        nodata=255,  # The value to use for missing data
    )

    # Write the predicted chip to the dataset
    dataset.write(predicted_chip[0].detach().cpu().numpy())

    # Return the georeferenced dataset
    return dataset


def georreferenced_chip_generator(dataloader, model, crs, pixel_size):
    # Source:
    # https://www.kaggle.com/code/luizclaudioandrade/torchgeo-101#Making-predictions,-applying-the-correct-georrefence-to-them-and-merging-them-in-one-file-that-is-saved-to-disk.

    """
    Generate georeferenced prediction chips.

    This function takes a dataloader, a model, a coordinate reference system (CRS), and a pixel size.
    It uses the model to make predictions for the data in the dataloader, and then georeferences the predictions
    using the CRS and pixel size. The georeferenced predictions are returned as a list.

    Parameters:
        dataloader (torch.utils.data.Dataloader): The dataloader containing the data to be predicted.
        model (segmentation_models.pytorch model): The model to use for making predictions.
        crs (str): The coordinate reference system to use for georeferencing the predictions.
        pixel_size (float): The size of each pixel in the predictions, in the units of the CRS.

    Returns:
        list: A list of georeferenced prediction chips.
    """
    # Initialize an empty list to hold the georeferenced prediction chips
    georref_chips_list = []

    # Iterate over the data in the dataloader
    for i, sample in enumerate(dataloader):
        # Extract the image, ground truth mask, and bounding box from the sample
        image, _, bbox = sample["image"], sample["mask"], sample["bbox"][0]

        # Use the model to make a prediction for the image
        prediction = model(image.cuda())

        # Create a geotransform for the prediction using the bounding box and pixel size
        geotransform = from_origin(bbox.minx, bbox.maxy, pixel_size, pixel_size)

        # Create a georeferenced chip from the prediction and add it to the list
        georref_chips_list.append(
            create_in_memory_geochip(prediction, geotransform, crs)
        )

    # Return the list of georeferenced prediction chips
    return georref_chips_list


def custom_avg(merged_data, new_data, merged_mask, new_mask, overlap, **kwargs):
    # https://gis.stackexchange.com/questions/469972/treat-overlapping-pixel-when-doing-merge-geotiff-images-using-rasterio
    # https://github.com/rasterio/rasterio/blob/main/rasterio/merge.py#L84
    # merged_data - array to update with new_data
    # new_data - data to merge same shape as merged_data
    # merged_mask, new_mask - boolean masks where merged/new data pixels are invalid same shape as merged_data

    # Creates an array with the same shape as merged_mask and True/False values randomly assigned
    mask = np.empty_like(merged_mask, dtype="bool")
    # Edit mask so that it is true if either merged_mask or new_mask is true
    # Mask will be True where the data is invalid
    np.logical_or(merged_mask, new_mask, out=mask)
    # Change mask so that True becomes False and False becomes True
    # Thus, mask is true where data is valid
    np.logical_not(mask, out=mask)

    # Create a weights array with ones
    weights = np.ones(mask.shape)

    for i in range(overlap - 1, 0 - 1, -1):
        weights[:, -i - 1, :] = (i + 1) / overlap
        weights[:, :, i] = (i + 1) / overlap

    merged_data[mask] = np.rint(
        (1 - weights[mask]) * merged_data[mask] + weights[mask] * new_data.data[mask]
    ).astype(int)

    # Now mask will be a opposite to the new_mask i.e. True where new_mask is False and vice versa
    # Therefore, mask now shows where the new_data is valid and should be copied to merged_data
    np.logical_not(new_mask, out=mask)
    # Update the mask so that it is True if both merged_mask and mask are True
    # i.e. mereged_mask is True (invalid data) and new_mask is False (valid data)
    # This is done to ensure that the new_data is only copied to merged_data where it is valid
    # and merged_data is invalid
    np.logical_and(merged_mask, mask, out=mask)
    # Copy the new_data to merged_data where mask is True, i.e. where new_data is valid and merged_data is invalid
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")


def merge_georeferenced_chips(chips_list, output_path, overlap, bounds):
    # Source:
    # https://www.kaggle.com/code/luizclaudioandrade/torchgeo-101#Making-predictions,-applying-the-correct-georrefence-to-them-and-merging-them-in-one-file-that-is-saved-to-disk.
    """
    Merge a list of georeferenced chips into a single GeoTIFF file.

    Parameters:
        chips_list (list): A list of Rasterio datasets representing the georeferenced chips.
        output_path (str): The path where the merged GeoTIFF file will be saved.

    Returns:
        None
    """

    # Create a wrapper function that includes the overlap parameter
    def custom_avg_with_overlap(merged_data, new_data, merged_mask, new_mask, **kwargs):
        return custom_avg(merged_data, new_data, merged_mask, new_mask, overlap)

    # Merge the chips using Rasterio's merge function
    merged, merged_transform = merge(
        chips_list, method=custom_avg_with_overlap, nodata=255, bounds=bounds
    )

    # Calculate the number of rows and columns for the merged output
    rows, cols = merged.shape[1], merged.shape[2]

    # Update the metadata of the merged dataset
    # The metadata is copied from the first chip in the list, and then the height, width, and transform are updated
    merged_metadata = chips_list[0].meta
    merged_metadata.update(
        {"height": rows, "width": cols, "transform": merged_transform}
    )

    # Write the merged array to a new GeoTIFF file
    # The output file is opened in write mode ('w'), and the updated metadata is passed in
    with rasterio.open(output_path, "w", **merged_metadata) as dst:
        dst.write(merged)

    # Close all the chips in the list
    # This is important to free up system resources
    for chip in chips_list:
        chip.close()


if __name__ == "__main__":

    data_path_g = os.path.join(
        "/mnt/g/RISK_PAPER", "PS_predictions_HPC", "test_border_and_pattern"
    )
    data_path_d = os.path.join("/mnt/d", "ML_model_data_paper")

    ml_model_input_dir = os.path.join(data_path_d, "input")
    # os.makedirs(ml_model_input_dir,exist_ok=True)

    # Set the seed for reproducibility
    seed = 54321
    generator = torch.Generator().manual_seed(seed)

    # Define the patch size and batch size for the model
    patch_size = 64
    batch_size = 16

    overlap = 8

    # Load the trained model
    model_path = os.path.join(
        data_path_g, "ML_model", "seed_54321_20231214-203338.ckpt"
    )
    model = PixelwiseRegressionTask.load_from_checkpoint(checkpoint_path=model_path)

    # PS prediction #
    # Define the path for the PS prediction raster

    prediction_path = os.path.join(
        data_path_d, "merged_prediction_updated_stiching.tif"
    )
    bounds = (int(3), int(50), int(8), int(54))

    if os.path.exists(prediction_path):
        print("PS is already predicted for this tile \n")
    else:

        # Create RasterDataset objects for coherence and infrastructure map images
        imgs_coh = RasterDataset_imgs_coh(
            os.path.join(ml_model_input_dir, "global_coh_NL", "summer_vv_rho"),
            transforms=scale_coh,
        )
        imgs_infrmap = RasterDataset_imgs_infmap(
            os.path.join(ml_model_input_dir, "OSM_input_channel_v2")
        )

        # Combine the two RasterDataset objects
        imgs_input = imgs_coh & imgs_infrmap

        # Set the dataset to predict
        dataset_to_pred = imgs_input

        # Create a GridGeoSampler object for the dataset
        sampler_pred = GridGeoSampler(dataset_to_pred, patch_size, patch_size - overlap)

        # Create dataloader where each patch is one batch (needed for correct geotiff generation)
        dataloader_pred_b1 = DataLoader(
            dataset=dataset_to_pred,
            batch_size=1,
            sampler=sampler_pred,
            num_workers=0,
            collate_fn=stack_samples,
            generator=generator,
        )

        # Set the model to evaluation mode
        # disable randomness, dropout, etc...
        model.eval()

        # Get the pixel size and CRS of the dataset
        pixel_size = dataset_to_pred.res
        crs = dataset_to_pred.crs.to_epsg()

        # Generate prediction chips
        start = time.time()  # Start measuring the time
        chips_generator = georreferenced_chip_generator(
            dataloader_pred_b1, model, crs, pixel_size
        )
        print("The time taken to predict was: ", time.time() - start, flush=True)

        # Save the prediction chips as a geotiff
        start = time.time()  # Start measuring the time
        merge_georeferenced_chips(chips_generator, prediction_path, overlap, bounds)
        print(
            "The time taken to generate a georrefenced image and save it was: ",
            time.time() - start,
            flush=True,
        )
