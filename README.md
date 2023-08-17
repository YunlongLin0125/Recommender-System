# Recommender Systems: looking further into the future
This is an implementation of proposed window-based prediction method.



Progress about my dissertation project, welcome to feedback and suggestion.
Contact: s2406121@ed.ac.uk

## folder structure

- **Recommender-System/**
  - **models/**
    - **SASRec/**
      - **data/**
      - **F_experiments/**
      - **Formal_runner/**
      - `main.py`
      - `model.py`
      - `utils.py`
    - **T2VRec/**
      - **data/**
      - **F_experiments/**
      - **Formal_runner/**
      - `main.py`
      - `model.py`
      - `model_sasrec.py`
      - `utils.py`
  - **images/**
  - `diagram.ipynb`
  - `Formal_Exp.xlsx`
  - `README.md`
  - `progress_stages.md`

**images/**: all images in the dissertation draw by [draw.io](draw.io)

**data/**: data and data processsing method is here

**F_experiments/**: All Formal experiments running steps in detail

**Formal_runner/**: bash files to carry out the experiments in `A,B,C,D`

`main.py`: running progress
`model.py`: model architecture files
`utils.py`: support functions like data partition
`diagram.ipynb`: draw dissertation diagram
`Formal_Exp.xlsx` : all experiments result
`progress_stages.md`: progress steps like a diary

## Project Page:
https://dpmt.inf.ed.ac.uk/msc/project/6930



## Potential dataset:
RetailRocket: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

MovieLength:
https://grouplens.org/datasets/movielens/



## State-of-the-art approaches:
GRU4Rec: [code](https://github.com/hidasib/GRU4Rec), [pytorch_version](https://github.com/hungthanhpham94/GRU4REC-pytorch), [paper](https://arxiv.org/abs/1511.06939)

SASRec: [code](https://github.com/kang205/SASRec), [pytorch_version](https://github.com/pmixer/SASRec.pytorch),  [paper](https://arxiv.org/abs/1808.09781)

BERT4Rec: [code](https://github.com/FeiSun/BERT4Rec), [paper](https://arxiv.org/abs/1904.06690)



## Evaluation metrics:

Same as [PinnerFormer](https://arxiv.org/abs/2205.04507)

* R@k: for Recommendation accuracy
* P90 coverage@k: P90 coverage means the smallest item sets that appear in the top k lists of at least 90% of the users.



## Parameters
You can try different models by changing the parameter:

### Model

#### Normal SASRec
`
--model=normal_sasrec
`
#### SASRec_Window
`
--model=sasrec_sampled
`
#### All Action
`
--model=all_action
`

#### Dense All Action
`
--model=dense_all_action
`


#### Combined

`--model=integrated`

#### Dense all action plus
`
--model=dense_all_action_plus
`

#### Combined dense all action plus
`
--model=dense_all_action_plus_plus`



### eval_epoch

`eval_epoch = 20`: evaluate each 20 epochs



### num_epochs

`num_epochs = 1000`:  set the limit of epoch number



### log_dir

`log_dir=test`:  create a folder `test` to record the training step

### loss_function
`loss_function=bce`
or
`loss_function=sampled_softmax`



### `--frozen_item=true --load_emb=true`

start transfer learning: we have using the embedding in **F_experiments/**



## Model Train (Time-Unaware models)

```
cd models/SASRec
```


### Train from Scratch on MovieLens-1M dataset (All action)

```
python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=bce --log_dir=test --model=all_action
```



### Transfer learning  on MovieLens-1M dataset (All action)

```
python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=test --model=all_action --load_emb=true --frozen_item=true
```



### Train from Scratch on MovieLens-20M dataset (All action)

```
python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=test --temporal=true --model=all_action
```



### Transfer learning  on MovieLens-20M dataset (All action)

```
python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=test --temporal=true --model=all_action --frozen_item=true --load_emb=true
```





## Model Train (Time-Aware models)

```
cd models/T2VRec
```
### Train from Scratch on MovieLens-20M dataset (T-All action)

```
python main.py --dataset=ml-20m --log_dir=test --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200
```

### Transfer learning  on MovieLens-20M dataset (T-All action)

```
python main.py --dataset=ml-20m --log_dir=test --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true
```



## Experiments Results (All models)

[Results](Formal_Exp.xlsx)
