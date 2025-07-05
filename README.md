# Global Geo-hazard Risk Assessment of Long-Span Bridges Enhanced with InSAR Availability
[![DOI](https://zenodo.org/badge/958456340.svg)](https://doi.org/10.5281/zenodo.15814218)

This repository contains code required to reproduce results for the paper entitled "Global Geo-hazard Risk Assessment of Long-Span Bridges Enhanced with InSAR Availability" (currently in review)

## Installation

Use Poetry to install Python environment with required dependencies as specified in poetry.lock and pyproject.toml (see: [Installing with poetry.lock](https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock))

```bash
poetry install
```

The code also requires the osmconvert and AWS CLI to be installed for downloading the required datasets. Installation instructions can be found here: 
- [osmconvert installation](https://wiki.openstreetmap.org/wiki/Osmconvert) (note: install in /usr/bin/osmconvert or update osm_convert_path in predict_PS_for_region.ipynb)
- [aws cli installation](https://aws.amazon.com/cli/)


## Usage

1) PS predictions
- ps_predictions/predict_PS_dens.ipynb can be used to generate PS predictions for small regions
- However, as those can be quite time-consuming, subsequent steps make use of already produced PS predictions that can be downloaded from the data repository on Zenodo ([Global Geo-hazard Risk Assessment of Long-Span Bridges Enhanced with InSAR Availability - research data](10.5281/zenodo.15797030)) 


2) Risk calculations

   Note: It requires the long-span bridges database CSV file, available on request from the authors of the associated [paper](https://doi.org/10.1080/15732479.2019.1639773). The dataset should be placed in data/bridge_db.
   Other required datasets should be downloaded to the data folder from [zenodo repository](10.5281/zenodo.15797030).
   
- First, the raw bridge database should be processed with risk_assessment/bridges_db/process_bridges_db.py to identify bridge shapes. The processing might take quite a lot of time. 

```bash
../.venv/bin/python -m risk_assessment.bridges_db.process_bridges_db
```

- Then, once the bridge geometries are generated, they should be divided into segments with risk_assessment/bridges_db/divide_into_segments.py 

```bash
../.venv/bin/python -m risk_assessment.bridges_db.divide_into_segments 
```

- Next, zonal statistics regarding PS availability, Sentinel availability, hazards, exposure and vulnerability should be generated with:

```bash
 ../.venv/bin/python -m risk_assessment.bridges_db.get_zonal_stats_from_bridge_lines
```

- Finally, data can be analysed with the following code to produce final risk scores

```bash
 ../.venv/bin/python -m risk_assessment.risk_calculation.lsb_risk_analysis 
```

- To get source data for plots, the following should be run: 

```bash
 ../.venv/bin/python -m risk_assessment.risk_calculation.plots_source_data_generation 
```

3) Paper figures

To reproduce plots from the paper, one can use the src/plots_for_paper/generate_plots.ipynb notebook. It makes use of the plots' source data published in [zenodo repository](10.5281/zenodo.15797030), but the data can be reproduced with the risk calculation scripts.


## License

[MIT](https://choosealicense.com/licenses/mit/)
