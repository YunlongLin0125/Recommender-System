python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/train_from_scratch/normal_sasrec --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=200 --temporal=true --lr=0.003 --maxlen=200

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/train_from_scratch/all_action --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=200 --temporal=true --lr=0.003 --maxlen=200

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/train_from_scratch/dense_all_action --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=200 --temporal=true --lr=0.003 --maxlen=200

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/train_from_scratch/dense_all_action+ --model=dense_all_action_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=200 --temporal=true --lr=0.003 --maxlen=200

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/train_from_scratch/dense_all_action++ --model=dense_all_action_plus_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=200 --temporal=true --lr=0.003 --maxlen=200

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/train_from_scratch/combined --model=integrated --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=200 --temporal=true --lr=0.003 --maxlen=200
