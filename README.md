# Global Geo-hazard Risk Assessment of Long-Span Bridges Enhanced with InSAR Availability

This repository contains code required to reproduce results for the paper entitlted "Global Geo-hazard Risk Assessment of Long-Span Bridges Enhanced with InSAR Availability" (currently in review)

## Installation

Use Poetry to install Python environment with required dependencies as specified in poetry.lock and pyproject.toml (see: [Installing with poetry.lock](https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock))

```bash
poetry install
```

The code requires also the osmconvert and aws cli to be installed for downloading the required datasets. Installation instructions can be found here: 
- [osmconvert installation](https://wiki.openstreetmap.org/wiki/Osmconvert) (note: install in /usr/bin/osmconvert or update osm_convert_path in predict_PS_for_region.ipynb)
- [aws cli installation](https://aws.amazon.com/cli/)


## Usage

1) PS predictions
- ps_predictions/predict_PS_dens.ipynb can be used to generate PS predictions for small regions
- however, as those can be quite time consuming, subsequent steps make used of already produced PS predictions that can be download from data repository on zenodo ([Global Geo-hazard Risk Assessment of Long-Span Bridges Enhanced with InSAR Availability - research data](10.5281/zenodo.15797030)) 


2) Risk calculations
- first, the raw bridge database should be processed with risk_assessment/bridges_db/process_bridges_db.py to identify bridge shapes. It requires the long-span bridges database csv file, available on request from authors of the associated [paper](https://doi.org/10.1080/15732479.2019.1639773). The dataset should be placed in data/bridge_db. The processing might take quite a lot of time. 

'''python
../.venv/bin/python -m risk_assessment.bridges_db.process_bridges_db
'''

- then, once the bridge geometries are generated, they should be divided into segments with risk_assessment/bridges_db/divide_into_segments.py 

'''python
../.venv/bin/python -m risk_assessment.bridges_db.divide_into_segments 
'''

- next, zonal statistics regarding PS avaialbility, Sentinel availability, hazards, exposure and vulnerability should be generated with:

'''python
 ../.venv/bin/python -m risk_assessment.bridges_db.get_zonal_stats_from_bridge_lines
'''

- finally, data can be analysed with the following code to produce final risk scores

'''python
 ../.venv/bin/python -m risk_assessment.risk_calculation.lsb_risk_analysis 
'''

- to get source data for plots the following should be run: 

'''python
 ../.venv/bin/python -m risk_assessment.risk_calculation.plots_source_data_generation 
'''

3) Paper figures

To reproduce plots from the paper, one can use the src/plots_for_paper/generate_plots.ipynb notebook. It makes use of plots source data published in [zenodo repository](10.5281/zenodo.15797030), but the data can be reproduced with the risk calculation scripts.


## License

[MIT](https://choosealicense.com/licenses/mit/)