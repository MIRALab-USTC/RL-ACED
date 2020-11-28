# Promoting Stochasticity for Expressive Policies via a Simple and Efficient Regularization Method

This repository is the official implementation of *Promoting Stochasticity for Expressive Policies via a Simple and Efficient Regularization Method*.

## Requirements

To install requirements:

```shell
pip install -r requirements.txt
```

## Training

1. You can edit the configuration files in the directory `configs`.

2. To train agent 

```shell
python scripts/run.py configs/aceb/aceb_gaussian.json --env_name HalfCheetah-v2
```

## Results

The results are saved in the directory `data`.