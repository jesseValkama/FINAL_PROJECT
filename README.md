# Light-weight Video Based Fall Detection

This is my bachelor's final project. 
Paper is available here (TODO:)

## Description

The project is a study (proposed by school) for fall detection with lightweight models (CNNs + RNNs). The Omnifall benchmark is used for evalution.
This project contains the training loop, the models, testing and inference. However, model weights are not released.

Results:

## Getting Started

### Dependencies

* PACKAGES: (TODO) 
* CUDA 12.6

### Installing

-

### Executing program (TODO:)

You might need to change some dependencies in the settings.yaml file
Also all of the training settings should be defined there
```
python main.py args
```
args:
    train: 0 | 1
    test: 0 | 1
    inference: 0 | 1

## Help

-

## Authors

Jesse Valkama

## Acknowledgments

MODELS
* [EFFICIENTNET](https://arxiv.org/pdf/1905.11946)
* [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)

DATASETS (omnifall annotations for le2i, more might be added at some point)
* [OMNIFALL](https://arxiv.org/abs/2505.19889)
* [LE2I](https://search-data.ubfc.fr/imvia/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html)

OTHER
* [README TEMPLATE](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)

