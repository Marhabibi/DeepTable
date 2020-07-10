# DeepTable

### A Permutation Invariant Neural Network for Table Orientation Classification

This repository contains a table corpus with their orientations annotations for tables from PMC, and additionally contains the code and the models for classifying tables into one the orientation layouts of horizontal, vertical, and matrix.
<br>
## Table preprocessing
## Train a model
To train a model, you need to use the DeepTableTrain.py and provide the number of epochs, the embedding initialization type (0: random initialization, 1: pretrained embeddings), the learning rate value, the location of the table data and the directory of the model:
```
DeepTableTrain.py [-h] -e N_EPOCH -v EMBEDDINGS_TYPE -l LR -i INPUT_TABLES -o MODEL_DIR
```


## Table Orientation tagging

The following command tags tables into one of three classes of horizontal, vertical and matrix. Before using this command you need to train a model.

```
DeepTableEval.py [-h] -m MODEL -i INPUT_TABLES -o OUTPUT_TAGS
```


## Citation
Please cite the following work
```
@article{habibi2020deeptable,
  title={A Permutation Invariant Neural Network for Table Orientation Classification},
  author={Habibi, Maryam and Starlinger, Johannes and Leser, Ulf},
  booktitle={Data Mining and Knowledge Discovery},
  year={2020}
}
```

