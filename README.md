# Recommender Systems: looking further into the future
## Overview
This is a time-aware transfer learning-based window predictor for recommender systems. The project is built and tested on a Windows desktop with specifications: RTX 3060 GPU, torch version 1.13.1, torchvision 0.14.1, and python 3.9.

[Project Page](https://dpmt.inf.ed.ac.uk/msc/project/6930)



## Table of Contents

- [Directory Structure](#directory-structure)
- [Datasets](#datasets)
- [State-of-the-Art Approaches](#state-of-the-art-approaches)
- [Evaluation Metrics](#evaluation-metrics)
- [Parameters](#parameters)
- [Model Training](#training)
  - [Model Train (Time-Unaware models)](#model-train-time-unaware-models)
  - [Model Train (Time-Aware models)](#model-train-time-aware-models)
- [Inference Only (Reproduce)](#inference-only-reproduce)
- [Experiment Results](#experiment-results)
- [Citation](#citation)



## Directory Structure

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

**data/**: data and data processsing method is here : `e.g. SASRec/data/processed/ml-1m.txt`or `e.g. T2VRec/data/ml-20m_train_withtime.txt`

**F_experiments/**: All Formal experiments results  in detail

**Formal_runner/**: bash files to carry out the formal experiments in `A,B,C,D`

`main.py`: running file, run the model.

`model.py`: model architecture files

`utils.py`: support functions like data partition

`diagram.ipynb`: draw dissertation diagram

`Formal_Exp.xlsx` : all experiment results

`progress_stages.md`: project steps like a diary

## Datasets
- **MovieLens**: [MovieLens Datasets](https://grouplens.org/datasets/movielens/). We use the 1M and 20M versions for testing.

  

## State-of-the-Art Approaches
- **SASRec**: 
  - **GitHub Repo**: [code](https://github.com/kang205/SASRec)
  - **PyTorch Version**: [pytorch_version](https://github.com/pmixer/SASRec.pytorch)
  - **Paper**: [paper](https://arxiv.org/abs/1808.09781)
  - **Description**: The first transformer-based sequential recommender system. Used as a baseline in our project.



## Evaluation Metrics

Metrics used are similar to those in [PinnerFormer](https://arxiv.org/abs/2205.04507):
- **Recall@k**: Measures recommendation accuracy.
- **P90 coverage@k**: Represents the smallest item sets that appear in the top-k lists of at least 90% of users.

In training step, it is normal to see (recall, 0.66) because we do not evaluate the diversity in validation.

## Parameters
- **Model Parameters**:
  - Normal SASRec: `--model=normal_sasrec`
  - SASRec_Window: `--model=sasrec_sampled`
  - All Action: `--model=all_action`
  - Dense All Action:  `--model=dense_all_action`
  - Combined: `--model=integrated`
  - Dense all action plus:`--model = dense_all_action_plus`
  - Combined dense all action plus: `--model=dense_all_action_plus_plus`
- **Evaluation Epoch**: `eval_epoch = 20`
- **Total Epochs**: `num_epochs = 1000`
- **Logging Directory**: `log_dir=test`
- **Loss Functions**: `loss_function=bce` or `loss_function=sampled_softmax`
- **Device**: Change to `cpu` if no GPU is available: `--device=cpu`
- **Item embedding freezing &  load item embedding**: `--frozen_item=true --load_emb=true` (transfer learning)



## Training
### Model Train (Time-Unaware models)
- Change to correct directory: `cd models/SASRec`
- **Training Commands (Some instances)**:
  
  - **Train from Scratch on MovieLens-1M dataset (All action)**: 
  
    ```bash
    python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=bce --log_dir=test --model=all_action
    ```
  - **Transfer learning  on MovieLens-1M dataset (All action)**: 
  
    - Transfer Learning approach will use the item embedding in **F_experiments/**
  
  
    ```bash
    python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=test --model=all_action --load_emb=true --frozen_item=true
    ```
  
  - **Train from Scratch on MovieLens-20M dataset (All action)**: 
  
      ```bash
      python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=test --temporal=true --model=all_action
      ```
  
  - **Transfer learning  on MovieLens-20M dataset (All action)**: 
  
      ```bash
      python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=test --temporal=true --model=all_action --frozen_item=true --load_emb=true
      ```

### Model Train (Time-Aware models)

- Change to correct directory: `cd models/T2VRec`
- **Training Commands**:
  - **Train from Scratch on MovieLens-20M dataset (T-All action)**:
    ```bash
    python main.py --dataset=ml-20m --log_dir=test --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200
    ```
  
  - **Transfer Learning on MovieLens-20M dataset (T-All action)**:
  
    ```bash
    python main.py --dataset=ml-20m --log_dir=test --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true
    ```
    

## Inference Only (Reproduce)

- For evaluating pre-trained model performance directly, add:
  - `--inference_only=true`
  - Example of loading a pre-trained model: `--state_dict_path=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/all_action/1/all_action.best.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth`

  
  
- **Training Commands**:

  - Change to correct directory:`cd models/SASRec` or `cd models/T2VRec`

  - **Time Unaware model (All action)**:

    ```bash
    python main.py --dataset=ml-20m --log_dir=test --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --inference_only=true --state_dict_path=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/all_action/1/all_action.best.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth
    ```

  - **Time Aware model (T-all action)**:

    ```bash
    python main.py --dataset=ml-20m --log_dir=test --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --inference_only=true --state_dict_path=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_all_action/1/all_action.best.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth
    ```



## Experiment Results
![Results Preview](images/data_preview.png)

[Detailed Results](Formal_Exp.xlsx)



## Citation

Our approach extends the SASRec model. More details can be found in the author's [repository](https://github.com/kang205/SASRec). Citation:
```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```

