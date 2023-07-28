cd ..

# ------------------------------------------------normal_sasrec
python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_normal_sasrec/1 --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=1

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_normal_sasrec/2 --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=2

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_normal_sasrec/3 --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=3

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_normal_sasrec/4 --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=4

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_normal_sasrec/5 --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=5

# ------------------------------------------------all_action
python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_all_action/1 --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=1

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_all_action/2 --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=2

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_all_action/3 --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=3

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_all_action/4 --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=4

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_all_action/5 --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=5

# ------------------------------------------------dense_all_action
python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/1 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=1

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/2 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=2

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/3 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=3

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/4 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=4

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/5 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=5


# ------------------------------------------------integrated
python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_integrated/1 --model=integrated --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=1

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_integrated/2 --model=integrated --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=2

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_integrated/3 --model=integrated --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=3

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_integrated/4 --model=integrated --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=4

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_integrated/5 --model=integrated --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=5


# ------------------------------------------------dense_all_action_plus
python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action_plus/1 --model=dense_all_action_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=1

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action_plus/2 --model=dense_all_action_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=2

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action_plus/3 --model=dense_all_action_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=3

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action_plus/4 --model=dense_all_action_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=4

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action_plus/5 --model=dense_all_action_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=5


# ------------------------------------------------dense_all_action_plus_plus
python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action_plus_plus/1 --model=dense_all_action_plus_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=1

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action_plus_plus/2 --model=dense_all_action_plus_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=2

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action_plus_plus/3 --model=dense_all_action_plus_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=3

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action_plus_plus/4 --model=dense_all_action_plus_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=4

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action_plus_plus/5 --model=dense_all_action_plus_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --load_emb=true --frozen_item=true --k_fold=5