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

## 628 (Wed.)

```
1. improve the model using teammate's method sampling strategy (change the train_target window)
2. try something transfer learning, item embedding for 1 model, 1 model for the prediction progress.
```

