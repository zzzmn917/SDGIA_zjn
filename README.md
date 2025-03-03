# SDGIA

## Code

This is the source code for DASFAA 2025 Paper.

## Requirements

- Python 3
- PyTorch >= 1.3.0
- tqdm

## Usage

Data preprocessing:

The code for data preprocessing can refer to [SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN).

Train and evaluate the model:
~~~~
python build_graph.py --dataset diginetica --sample_num 12
python main.py --dataset diginetica
~~~~

