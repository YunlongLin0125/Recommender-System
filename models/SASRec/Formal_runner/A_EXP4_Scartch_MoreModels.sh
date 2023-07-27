cd ..

#----------------------------------combined model
# python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/integrated/1 --model=integrated

# python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/integrated/2 --model=integrated

# python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/integrated/3 --model=integrated

# python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/integrated/4 --model=integrated

# python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/integrated/5 --model=integrated


#----------------------------------dense all +
# python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/dense_all_action_plus/1 --model=dense_all_action_plus

# python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/dense_all_action_plus/2 --model=dense_all_action_plus

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/dense_all_action_plus/3 --model=dense_all_action_plus

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/dense_all_action_plus/4 --model=dense_all_action_plus

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/dense_all_action_plus/5 --model=dense_all_action_plus


#----------------------------------dense all ++
python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/dense_all_action_plus_plus/1 --model=dense_all_action_plus_plus

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/dense_all_action_plus_plus/2 --model=dense_all_action_plus_plus

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/dense_all_action_plus_plus/3 --model=dense_all_action_plus_plus

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/dense_all_action_plus_plus/4 --model=dense_all_action_plus_plus

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=20 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.001/dense_all_action_plus_plus/5 --model=dense_all_action_plus_plus