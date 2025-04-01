# Import modules

import matplotlib.pyplot as plt

import torch

from torchgeo.datasets import RasterDataset


# Define functions and classes #
# Define classes for raster datasets with images and masks
class RasterDataset_imgs(RasterDataset):
    filename_glob = "*.tif"
    is_image = True


class RasterDataset_msks(RasterDataset):
    filename_glob = "*.tif"
    is_image = False


# Define class for reading optical data as input for plots
class RasterDatasetOptical_imgs(RasterDataset):
    filename_glob = "*.tif"
    is_image = True
    separate_files = False
    all_bands = ["R", "G", "B", "NIR"]
    rgb_bands = ["R", "G", "B"]

    def plot(self, sample):
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image
        image = sample["image"][rgb_indices].permute(1, 2, 0)
        #         image = sample["image"][rgb_indices]
        image = torch.clamp(image / 1, min=0, max=1).numpy()

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig


# Define function for scaling coherence data and converting it from int to float
def scale_coh(item: dict):
    item["image"] = item["image"] / 1000.0
    return item


# Function for ploting the input images and mask
def plot_input(dataset: dict, rows: int = 1, width: int = 10):
    img = dataset["image"]
    mask = dataset["mask"]

    nb_input_chnls = img.shape[0]

    if rows == 1:
        # create a grid
        _, axs = plt.subplots(
            1, nb_input_chnls + 1, figsize=(nb_input_chnls + 1 * width, 1 * width)
        )

        for i in range(nb_input_chnls):
            axs[i].imshow(img[i].squeeze().numpy(), cmap="turbo", vmin=0, vmax=1)
            axs[i].axis("off")

        axs[nb_input_chnls].imshow(
            mask.squeeze().numpy(), cmap="turbo", vmin=0, vmax=60
        )
        axs[nb_input_chnls].axis("off")
    else:
        columns = int((nb_input_chnls + 1) / rows + (nb_input_chnls + 1 % rows > 0))
        _, axs = plt.subplots(rows, columns, figsize=(columns * width, rows * width))

        r = 0
        c = 0

        for i in range(nb_input_chnls):
            axs[r][c].imshow(img[i].squeeze().numpy(), cmap="turbo", vmin=0, vmax=1)
            axs[r][c].axis("off")

            if c < (columns - 1):
                c = c + 1
            else:
                c = 0
                r = r + 1

        axs[r][c].imshow(mask.squeeze().numpy(), cmap="turbo", vmin=0)
        axs[r][c].axis("off")
