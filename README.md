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

Same as [PinnerFormer](#https://arxiv.org/abs/2205.04507)

* R@k: for Recommendation quality
* P90 coverage@k: P90 coverage means the smallest item sets that appear in the top k lists of at least 90% of the users.

# Test Code:

## SASRec
### movieLens-1M (SASRec):

![image-20230612134712835](C:\Users\Yunlong\AppData\Roaming\Typora\typora-user-images\image-20230612134712835.png)

![image-20230612134813391](C:\Users\Yunlong\AppData\Roaming\Typora\typora-user-images\image-20230612134813391.png)

```
cd models/SASRec
python main.py --dataset=processed/ml-1m --train_dir=test --maxlen=200 --dropout_rate=0.2 --device=cuda --window_predictor=false --window_eval=false
```

### Ml-1M (SASRec But evaluate on our evaluation metrics)

![image-20230612134801457](C:\Users\Yunlong\AppData\Roaming\Typora\typora-user-images\image-20230612134801457.png)

```
cd models/SASRec
python main.py --dataset=processed/ml-1m --train_dir=test --maxlen=200 --dropout_rate=0.2 --device=cuda --window_predictor=false --window_eval=true
```

### Ml-1M (SASRec change the input feeding strategy and evaluate on our evaluation metrics)

![image-20230612134731595](C:\Users\Yunlong\AppData\Roaming\Typora\typora-user-images\image-20230612134731595.png)

```
cd models/SASRec
python main.py --dataset=processed/ml-1m --train_dir=test --maxlen=200 --dropout_rate=0.2 --device=cuda --window_predictor=true --window_eval=true
```



### RetailRocket (Not tested):

```
cd models/SASRec
python main.py --dataset=retailrocket --train_dir="store_filepath" --maxlen=50 --dropout_rate=0.5 --device=cuda
```





[//]: # (## GRU4Rec &#40;Incomplete&#41;)

[//]: # (### movieLens-1M:)

[//]: # (```)

[//]: # (cd models/GRU4Rec)

[//]: # (python main.py --dataset=ml-1m_repro --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda)

[//]: # (```)

[//]: # (### RetailRocket:)

[//]: # (```)

[//]: # (cd models/GRU4Rec)

[//]: # (python main.py --dataset=retailrocket --train_dir=default --maxlen=50 --dropout_rate=0.5 --device=cuda)

[//]: # (```)
[#https://arxiv.org/abs/2205.04507]: 