cd ..


# ------------------------------------------------normal_sasrec
python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_normal_sasrec/1 --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_normal_sasrec/1 --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=2

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_normal_sasrec/1 --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=3

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_normal_sasrec/1 --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=4

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_normal_sasrec/1 --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=5

# ------------------------------------------------all_action
python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_all_action/1 --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_all_action/1 --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=2

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_all_action/1 --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=3

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_all_action/1 --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=4

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_all_action/1 --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=5

# ------------------------------------------------dense_all_action
python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/1 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/1 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=2

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/1 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=3

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/1 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=4

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/1 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=5


# ------------------------------------------------combined
python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/1 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/1 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=2

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/1 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=3

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/1 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=4

python main.py --dataset=ml-20m --log_dir=F_experiments/T/ml-20m/transfer/sampled_softmax/lr=0.001/t2v_dense_all_action/1 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=1000 --temporal=true --lr=0.001 --maxlen=200 --k_fold=5