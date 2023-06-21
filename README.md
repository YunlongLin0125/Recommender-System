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
GRU4Rec: [code](https://github.com/hidasib/GRU4Rec), [pytorch_version](https://github.com/hungthanhpham94/GRU4REC-pytorch), [paper](https://arxiv.org/abs/1511.06939)

SASRec: [code](https://github.com/kang205/SASRec), [pytorch_version](https://github.com/pmixer/SASRec.pytorch),  [paper](https://arxiv.org/abs/1808.09781)

BERT4Rec: [code](https://github.com/FeiSun/BERT4Rec), [paper](https://arxiv.org/abs/1904.06690)



## Evaluation metrics:

Same as [PinnerFormer](https://arxiv.org/abs/2205.04507)

* R@k: for Recommendation quality (Currently focus on)
* P90 coverage@k: P90 coverage means the smallest item sets that appear in the top k lists of at least 90% of the users.

# Test Code:

### Ml-1M (Normal SASRec)

<img src="materials\Window_eval.png" alt="Image" style="width:70%; height:auto;">



```
cd models/SASRec
python main.py --dataset=processed/ml-1m --log_dir=test --maxlen=200 --dropout_rate=0.2 --device=cuda --model=normal_sasrec --window_eval=true
```

### Ml-1M (SASRec sampled)



<img src="materials\window_split.png" alt="Image" style="width:70%; height:auto;">



<img src="materials\SASRec_changeinput.png" alt="Image" style="width:50%; height:auto;">

```
cd models/SASRec
python main.py --dataset=processed/ml-1m --log_dir=test --maxlen=200 --dropout_rate=0.2 --device=cuda --model=sasrec_sampled --window_eval=true
```

### Ml-1M (All action)

```
cd models/SASRec
python main.py --dataset=processed/ml-1m --log_dir=test --maxlen=200 --dropout_rate=0.2 --device=cuda --model=all_action--window_eval=true
```



### Ml-1M (Dense all action)

```
cd models/SASRec
python main.py --dataset=processed/ml-1m --log_dir=test --maxlen=200 --dropout_rate=0.2 --device=cuda --model=dense_all_action --window_eval=true
```



### RetailRocket (Normal SASRec):

```
cd models/SASRec
python main.py --dataset=processed/retailrocket --log_dir=test --maxlen=50 --dropout_rate=0.5 --device=cuda --model=normal_sasrec --window_eval=true
```


### RetailRocket (SASRec Sampled):

```
cd models/SASRec
python main.py --dataset=processed/retailrocket --log_dir=test --maxlen=50 --dropout_rate=0.5 --device=cuda --model=sasrec_sampled --window_eval=true
```

### RetailRocket (All action):

```
cd models/SASRec
python main.py --dataset=processed/retailrocket --log_dir=test --maxlen=50 --dropout_rate=0.5 --device=cuda --model=all_action --window_eval=true
```

### RetailRocket (Dense all action):

```
cd models/SASRec
python main.py --dataset=processed/retailrocket --log_dir=test --maxlen=50 --dropout_rate=0.5 --device=cuda --model=dense_all_action --window_eval=true
```


