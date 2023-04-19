# Recommender Systems: looking further into the future
Progress about my dissertation project, welcome to feedback and suggestion.
Contact: s2406121@ed.ac.uk


## Project Page:
https://dpmt.inf.ed.ac.uk/msc/project/6930

## Potential dataset:
RetailRocket: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset


MovieLength:
https://grouplens.org/datasets/movielens/

## State-of-the-art approaches:
GRU4Rec: [code](https://github.com/hidasib/GRU4Rec), [pytorch_code](https://github.com/hungthanhpham94/GRU4REC-pytorch) [paper](https://arxiv.org/abs/1511.06939)

SASRec: [code](https://github.com/kang205/SASRec),[pytorch_code](https://github.com/pmixer/SASRec.pytorch)  [paper](https://arxiv.org/abs/1808.09781)

BERT4Rec: [code](https://github.com/FeiSun/BERT4Rec), [paper](https://arxiv.org/abs/1904.06690)


# Test Code:
## SASRec
### movieLens-1M:
```
cd models/SASRec
python main.py --dataset=ml-1m_repro --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```
### RetailRocket:

```
cd models/SASRec
python main.py --dataset=retailrocket --train_dir=default --maxlen=50 --dropout_rate=0.5 --device=cuda
```

## GRU4Rec (Incomplete)
### movieLens-1M:
```
cd models/GRU4Rec
python main.py --dataset=ml-1m_repro --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```
### RetailRocket:
```
cd models/GRU4Rec
python main.py --dataset=retailrocket --train_dir=default --maxlen=50 --dropout_rate=0.5 --device=cuda
```