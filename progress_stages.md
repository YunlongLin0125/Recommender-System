# week 0

## 529 (Mon.)

``` 
1. set up eddie
2. test eddie usage with a script file (not GPU)
```

## 530 (Tues.)

```
1. prepare before restarting the project
```

## 531 (Wed.)

``` 
1. reread the previous progress of the project
2. reread the IPP report
3. reproduce the SASRec again
4. research the implementation of GRU4Rec
```

## 601 (Thur.)

```
1. reproduce GRU4Rec with recSys15 dataset (R@20:0.4833; mrr:0.1477)
2. test Eddie for GPU use (Pytorch)
```

## 604 (Sun.)

```
1. slides preparation
2. reproduce GRU4Rec's dataset splitting strategy
```

## 605 (Mon.)

```
1. slides preparation
2. Review what have done, look on what need to be done in the future
```

# Week 1

## 606 (Tues.)

```
1. try Recall@k evaluation metrix
2. understand P90 coverage@k
```

## 607 (Wed.)

```
1. Reread SASRec, Pinnerformer.
Why sampled softmax in Pinnerformer but the P90 coverage still works?
2. Try to implement P90 coverage@k.
```

### Q:

```
1. Sampled softmax vs. Full softmax.
SASRec only use 100 random samples, even in the evaluation, which will make a higher R@10. But during the inference, we need a full softmax to get a more accurate result, but SASRec still combine the groundtruth with 100 unifirm negative samples.
2. P90 coverage@10 in Pinnerformer.
'we compute embeddings at time 𝑡 for all users in the evaluation set, and then measure how well the embedding at time 𝑡 retrieves all Pins a user will engage with from time 𝑡 to 𝑡 + 14𝑑 from an index of 1M random Pins. Assuming we have a set of users, U, a set of positively engaged Pins P𝑈 for each user 𝑈 , and a random corpus of 1M Pins N'
the positive engagements includes in 1M pins N or not?
=> 
positive engaged Pins P + negative random sampling = 1M 
OR
randoms corpus = 1M. (currently implemented) we don't consider whether the groundtruth items included or not.
Result in different P90 Coverage@10.

what fraction of the index of 1M Pins accounts for 90% of the top 10 retrieved results over a set of users ("P90 Coverage@10").

```

## 608 (Thur.)

```
1. Change the data feeding in SASRec.
2. Split the dataset for the window-based predictor.
3. Implement a easy model can handle the window-based predictor (modification on the data reading step of the SASRec code)
```

## 609 (Fri.)

```
1. Realise the difference of training objective between this change data feeding strategy with all action prediction
2. Currentlys work as window-predictor split the training target by percentage, but still evaluate on the next item prediction. Evaluation metrics: P90 coverage@k (recommending diversity), (R@k)recommender quality. (Following PinnerFormer)
3. Plan to change the evaluation on predicting all items at percentage
```

## 612 (Mon.)

```
1. Merge the code to the main branch
2. Update the Readme File
3. Slides prepare
4. Code refactor
5. PopRec model
```

# Week 2

## 614 (Wed.)

```
1. Try implement all action prediction method.
```

## 615 (Thur.)
```
1. Implement Evaluation metrics on HT and nDCG, independently evaluate each val/test target.
```

## 616 (Fri.)
```
1. Change back to primary Recall@k Evaluation (SASRec > SASRec-window > all actions).
2. the reason for action may be the signal flow is too low? (not sure). Adding the negative sampling may improve. 
(not sure, teammates say not).
```

## 618 (Sun.)

```
1. refactor all action prediction
2. implement dense all action
```

## 619 (Mon.)

```
1. try all models on retailrocket and record the results (focus on recall now)
2. prepare for the presentation slides (4 training objectives/ Results Analysis)
```

# Week 3

## 621 (Wed.)

```
1. Code refactor
2. Change the loss function (potentially)
```

## 622 (Thur.)

```
1. Implement the loss function Sampled softmax loss only for SASRec now.
```

## 625 (Sun.)

```
1. Implement loss function for all models.
2. Plan to find the optimal hyperparameters for different models.
3. Implement Dense all +, Dense all ++, and a combined model.
```

## 626 (Mon.)

```
1. Run experiments for all existing models.
2. Slides preparation
3. One team mate(U) says his all action started working(means got better result than baseline?)
4. Experiments(hyperparameters for all action model)
5. Experiments(Compared the loss function Sampled Softmax v.s. BCE)
6. Experiments(new models on ml-1m with sampled softmax loss)
```

# Week 4

## 628 (Wed.)

```
1. improve the model using teammate's method sampling strategy (change the train_target window)
2. try something transfer learning, item embedding for 1 model, 1 model for the prediction progress (other parameters).
## if the transfer learning still does not work. Need to check the implementation.
<<<<<<< HEAD
```

## 629 (Thur.)

```
1. Tried some models with transfer learning, the model is still unexpected.
2. Recheck the training objective of each model, make sure each model is trained as expected.
```


## 630 (Fri.)

```
1. Tried some models with transfer learning, the model is still unexpected.
2. Recheck the training objective of each model, make sure each model is trained as expected.
```



## 701 (Sat.)

```
1. RetailRocket has some positive results (only all action improve the normal sasrec with frozen item embedding)
2. The reason for this case: percentage window has more positive connections for the prediction parameters.
3. Assumption: If we change the percentage window to temporal window, Things will be better,
```


## 703 (Mon.)
```
1. Try ML-20M.
2. prepare slides
```

# Week 5

## 704 (Tue.)
```
1. check the loss function for all models. (some problems to avoid the influence of pad-value)
```

## 705 (Wed.)
```
1. Fix the problem for all of the loss function (avoid padding sequence position influence).
Except all action prediction (No need to mask)
2. Run all proposed model get the experiments result
3. Try fine tuning (smaller lr for item embedding)
```

## 706 (Thur.)
```
1. refactor the directory to split the percentage window and temporal window.
2. how to obtain a better item embedding. (? more data beyond the temporal window)
```
## Time Feature
## 707 (Fri.)

```
Read → Understand
1. read tisasrec, incorparating sasrec with temporal features.
2. Reproduce tisasrec results.
```

## 709 (Sun.)
```
Understand → Implement
1. Understand the code (Relation Matrix)
2. implement TiSASRec window predictor (for all action)
```

## 710 (Mon.)

```
1. Not finish the implementation for window predictor incorparating
2. Memory error (relation matrix)
3. Slides preparation
```

# week 6

## 711 (Tue.)

```
1. Progress report
2. Suggest & Read T2V, which just incorporate time as a feature, rather than time interval matrix (Memory cost).
```

## 712  (Wed.)

```
1. Progress report
2. Code Refactor.
3. Current Step: Try to implement transfer learning to TiSASRec (Haven't implement yet)
4. finish the progress report
```

## 713 (Thur.)

```
1. read paper time2vec
```

## 715 (Sat.)

```
1. Understand the paper time2vec.
Periodic activation (sine).
2. Understand what Pinnerformer is doing.
Periodic activations (cosine (P dims), sine(P dims) and log transformation (1 dim)) (2P + 1)
```

## 716 (Sun.)

```
1. implement time2vec (only one time feature, abs: time diff).
2. extend its to window-predictor.
```

## 717 (Mon.)

```
1. Run experiments (lr=0.003; eval_epo = 2; total_ep = 20) on t2v-models
Results are collected by best in valid dataset and correcsponding test
2. Rerun previous exepriments on normal models
3. slides preparation
```

# week 7

## 718 (Tue.)

```
1. figure out the abs timestamp and rel timestamp
2. try implement the timeseqs include two features
```



## 719 (Wed.)

```
Clarify Pinnerformer:
1. raw timestamp(abs) 25 dims
2. Time Difference(rel) 65 dims
3. Time Gap(rel) 65 dims

Train a normal sasrec from scratch (not well performed)
```



## 720 (Thur.)

```
1. Run pinnerformer struture (time features[raw, time_diff, time_gap]) for transfer learning.
2. try replace raw timestamps with scaled raw timestamps
3. Figure out what need to be done, run formal experiments.
```



## 721 (Fri.)

```
1. P90 Coverage@10 (makes the training very slow)
2. Early stopping by validation loss (Not complete)
2. Cross Validation (Not complete)
```



## 724 (Mon.)

```
1. Implement early stop to all models (Based on Validation Recall only)
2. Think about the way better evaluate all models fairly (Cross Validation)

but you need to make sure all models test on the same dataset.
Same way to create test set.

Or k fold cross-validation
Validation + Train shuffle with the same seed across all the models, then create k model performance.

manually get average

3. Table Making
```



# Week 8

## 725 (Tue.)

```
1. Code Refactor
2. Make sure all formal experiments (Table Making)
```



## 726 (Wed.) 

## Experiments Log
```
EXP1.
pos, neg
sasrec(1, 1)
sasrec-sampled (1, 1)
Window predicts (targets, 10 negs)
```





```
1. Experiment Results Collection (A_EXP1, A_EXP2, A_EXP3, A_EXP4)
2. Code Preparing for temporal K-fold CrossValidation
```



## 727 (Thur.)

```
1. Code Running
2. first part of introduction
```



## 728 (Fri.)

```
1. Exp Running
2. Eddie
3. Introduction
```



## 730 (Sun)

```
1. finish the draft writing for introduction part.
2. run exp.
```



## 731 (Mon.)

```
1. complete all experiments and get results written on excel file.
2. 
```



## Q

```
1. recommendation accuracy = Recall

```



# Formal Experiments (Carry out from 7.25)

```
1. Next item prediction vs window predictor (Show the problem of window-predictor)____Train from Scratch
1.5 *(SASRec-sampled)
2. Increase the singal intensity in the network (window-predictor trainining objective) _____Train From Scratch
3. BCE vs. Sampled Softmax Loss (Loss function choice)_____Train From Scratch
4. Learning Rate (learning rate choice)_____Train From Scratch
5. train from scratch vs. transfer learning (Method proposed inspired by supervisor and Pinnerformer)
6. data split by percentage vs. split by time (window-choice)
7. time-ignorance vs. t2v (influence of time features)
8. Timestamps encoding. (time features choice)
```



Main Eval Task: How well each can predict all future purchases over the next k days

```
1. TFS: next item prediction (SASRec) vs. percentage window predictor

* Increase the signal intensity (new models proposed)
* Loss function Selected
* negative samples selection
* Hyperparameter tuning (learning rate)

↓ Results leads to transfer learning ↓

2. TL: next item vs. percentage window predictor
3. TL & TFS： next item predictor vs. temporal window predictor
4. TL & TFS: Temporal window predictor vs. Ti temoral window predictor
```





# TODO

1. <span style="color:red">incorporate relative time features </span>
2. <span style="color:red">P90 coverage evaluation metric</span>
3. Formal experiments run
4. Dissertation
5. Hyperparameter tuning (optional)
6. <span style="color:blue">Try different data splitting strategy (optional)</span>
7. More datasets (optional)
8. Check the code working of percentage window.
