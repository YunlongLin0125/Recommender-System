cd ..

# ------------------------------------------------normal_sasrec
python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/normal_sasrec/1 --temporal=true --model=normal_sasrec --k_fold=1 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/normal_sasrec/2 --temporal=true --model=normal_sasrec --k_fold=2 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/normal_sasrec/3 --temporal=true --model=normal_sasrec --k_fold=3 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/normal_sasrec/4 --temporal=true --model=normal_sasrec --k_fold=4 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/normal_sasrec/5 --temporal=true --model=normal_sasrec --k_fold=5 --frozen_item=true --load_emb=true

# ------------------------------------------------All Action
python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/all_action/1 --temporal=true --model=all_action --k_fold=1 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/all_action/2 --temporal=true --model=all_action --k_fold=2 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/all_action/3 --temporal=true --model=all_action --k_fold=3 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/all_action/4 --temporal=true --model=all_action --k_fold=4 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/all_action/5 --temporal=true --model=all_action --k_fold=5 --frozen_item=true --load_emb=true

# ------------------------------------------------Dense All Action

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action/1 --temporal=true --model=dense_all_action --k_fold=1 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action/2 --temporal=true --model=dense_all_action --k_fold=2 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action/3 --temporal=true --model=dense_all_action --k_fold=3 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action/4 --temporal=true --model=dense_all_action --k_fold=4 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action/5 --temporal=true --model=dense_all_action --k_fold=5 --frozen_item=true --load_emb=true


# ------------------------------------------------Combined

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/integrated/1 --temporal=true --model=integrated --k_fold=1 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/integrated/2 --temporal=true --model=integrated --k_fold=2 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/integrated/3 --temporal=true --model=integrated --k_fold=3 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/integrated/4 --temporal=true --model=integrated --k_fold=4 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/integrated/5 --temporal=true --model=integrated --k_fold=5 --frozen_item=true --load_emb=true

# ------------------------------------------------Dense All Action + 

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus/1 --temporal=true --model=dense_all_action_plus --k_fold=1 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus/2 --temporal=true --model=dense_all_action_plus --k_fold=2 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus/3 --temporal=true --model=dense_all_action_plus --k_fold=3 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus/4 --temporal=true --model=dense_all_action_plus --k_fold=4 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus/5 --temporal=true --model=dense_all_action_plus --k_fold=5 --frozen_item=true --load_emb=true

# ------------------------------------------------Dense All Action ++

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus_plus/1 --temporal=true --model=dense_all_action_plus_plus --k_fold=1 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus_plus/2 --temporal=true --model=dense_all_action_plus_plus --k_fold=2 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus_plus/3 --temporal=true --model=dense_all_action_plus_plus --k_fold=3 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus_plus/4 --temporal=true --model=dense_all_action_plus_plus --k_fold=4 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus_plus/5 --temporal=true --model=dense_all_action_plus_plus --k_fold=5 --frozen_item=true --load_emb=true