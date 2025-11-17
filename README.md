# Deep-Learning-Playground
A space for experimenting with deep learning.


# Installation
+ Docker image as in test.sh or test.ps1

+ Container for testing
```bash
bash env.sh
source ~/.bashrc
```

+ Directly use
```bash
pip install -e .
```


# Quick start
+ Tabular single regression
    + Dataset: [Boston Housing](https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset/data)
    + Descriptions:
        + Dense Neural Network
        + (Batch, 12) -> (Batch, 1)
    + Commands:
```bash
python _examples/boston_housing/preprocess.py
python main.py --mode train --cfg _examples/boston_housing/cfg.yaml
python main.py --mode valid --cfg _examples/boston_housing/cfg.yaml
python main.py --mode infer --cfg _examples/boston_housing/cfg.yaml
```

+ Tabular time series multiple regression
    + Dataset: [Stock Pricing Dataset](https://www.kaggle.com/datasets/hershyandrew/amzn-dpz-btc-ntfx-adjusted-may-2013may2019)
    + Descriptions:
        + Long-Short-Term-Memory or TransformerEncoder
        + default window size = 64
        + autoregression generated dataset
        + (Batch, 64, 4) -> (Batch, 4)
    + Commands
```bash
python _examples/stock_pricing/preprocess.py
python main.py --mode train --cfg _examples/stock_pricing/cfg.yaml
python main.py --mode valid --cfg _examples/stock_pricing/cfg.yaml
python main.py --mode infer --cfg _examples/stock_pricing/cfg.yaml
```

+ Image classification
    + Dataset: [Mnist](https://www.kaggle.com/c/dogs-vs-cats/data?select=train.zip)
    + Description
        + Convolutiontal Nweural Network or ResNet18 or Vit_b_16
        + (Batch, 1, 28, 28) -> (Batch, 10)
    + Commands
```bash
python _examples/mnist/preprocess.py
python main.py --mode train --cfg _examples/mnist/cfg.yaml
python main.py --mode valid --cfg _examples/mnist/cfg.yaml
python main.py --mode infer --cfg _examples/mnist/cfg.yaml
```


# Design
+ modularization, reusability, extensibility, flexibility, Readability
+ see [main.py](./main.py) and [configs](./_examples/boston_housing/cfg.yaml) to view the workflow of this framework
