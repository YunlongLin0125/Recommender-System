cd ../..

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.005 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.005/sasrec_sampled/1 --model=sasrec_sampled

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.005 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.005/sasrec_sampled/2 --model=sasrec_sampled

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.005 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.005/sasrec_sampled/3 --model=sasrec_sampled

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.005 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.005/sasrec_sampled/4 --model=sasrec_sampled

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.005 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.005/sasrec_sampled/5 --model=sasrec_sampled


python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.0005 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.0005/sasrec_sampled/1 --model=sasrec_sampled

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.0005 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.0005/sasrec_sampled/2 --model=sasrec_sampled

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.0005 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.0005/sasrec_sampled/3 --model=sasrec_sampled

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.0005 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.0005/sasrec_sampled/4 --model=sasrec_sampled

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=1000  --lr=0.0005 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/Scratch/sampled_softmax/lr=0.0005/sasrec_sampled/5 --model=sasrec_sampled