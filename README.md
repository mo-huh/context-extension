# lra-benchmarks

This repository is based on the Long Range Arena (LRA) datasets from Google Research. The original repository from Google is available [here](https://github.com/google-research/long-range-arena). The paper is available on [ArXiv](https://arxiv.org/pdf/2011.04006.pdf).

In this repo we provide researchers with code for some of the simpler tasks. It also has the advantage of using PyTorch and HuggingFace Transformers instead of Jax/Flax, with which less are familiar. In theory, this (arguably) makes our code easier to understand and extend for some researchers.

## Use Benchmark
1. Run `sh ./get_data.sh`. It will create a new `datasets` directory in the project and will populate it with the required datasets (make sure to run this in the project's root directory as the script uses relative paths)
2. To test the code with simple models, just run: `python run_model.py`

_________________________________________________________________________

# IMDb Text Classification

This project fine-tunes pre-trained transformer models (e.g., BERT) for text classification on the IMDb dataset. The goal is to experiment with techniques to extend the context length of short-context models.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
2. Download the Dataset: Use the Python script:
   `python fetch_data.py --task imdb`
   Alternatively, use the shell script:
   `bash get_data.sh`


## Run the Code

1. Train and Evaluate: Execute the main script:
   `python run_model.py`
2. Modify Configuration: Adjust hyperparameters in lra_config.py (e.g., batch size, learning rate, or epochs).

