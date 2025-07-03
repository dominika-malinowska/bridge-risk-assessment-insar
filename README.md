# Global Geo-hazard Risk Assessment of Long-Span Bridges Enhanced with InSAR Availability

This repository contains code required to reproduce results for the paper entitlted "Global Geo-hazard Risk Assessment of Long-Span Bridges Enhanced with InSAR Availability" (currently in review)

## Installation

To install a Python environment, use Poetry, which will use the poetry.lock and pyproject.toml to install required dependencies (see: [Installing with poetry.lock](https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock))

```bash
poetry install
```

The code requires also the osmconvert and aws cli to be installed for downloading the required data. Installation instructions can be found here: 
- [osmconvert installation](https://wiki.openstreetmap.org/wiki/Osmconvert) (note: install in /usr/bin/osmconvert or update osm_convert_path in predict_PS_for_region.ipynb)
- [aws cli installation](https://aws.amazon.com/cli/)


## Usage

ps_predict/predict_PS_dens.ipynb can be used to generate PS predictions


## License

[MIT](https://choosealicense.com/licenses/mit/)