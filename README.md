## Getting Started
### Prerequisites
- Python 3.10+
### Installation
- Init a new virtual environment
  - For conda, `conda env create -f full_environment.yml` and active it by `conda activate ai_subspace_env`
  - For venv, `py -m venv ai_subspace_env`
    - Activate the virtual environment
    - Install the required packages by running `pip install -r requirements.txt`

- For both cases, you'll need to install seperately a torch package (>=2.7.0) that matches your CUDA version,
  visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for more details.

- See main.py for configuring parameters and running the simulation.

