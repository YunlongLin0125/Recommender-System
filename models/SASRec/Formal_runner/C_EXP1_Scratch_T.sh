cd ..

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=100  --lr=0.001 --loss_function=sampled_softmax --log_dir=experiments/temporal/ml-20m/train_from_scratch/normal_sasrec  --model=normal_sasrec --frozen_item=false --temporal=true --finetune=false --load_emb=false

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=100  --lr=0.001 --loss_function=sampled_softmax --log_dir=experiments/temporal/ml-20m/train_from_scratch/all_action  --model=all_action --frozen_item=false --temporal=true --finetune=false --load_emb=false

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=100  --lr=0.001 --loss_function=sampled_softmax --log_dir=experiments/temporal/ml-20m/train_from_scratch/dense_all_action  --model=dense_all_action --frozen_item=false --temporal=true --finetune=false --load_emb=false

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=100  --lr=0.001 --loss_function=sampled_softmax --log_dir=experiments/temporal/ml-20m/train_from_scratch/dense_all_action_+  --model=dense_all_action_plus --frozen_item=false --temporal=true --finetune=false --load_emb=false

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=100  --lr=0.001 --loss_function=sampled_softmax --log_dir=experiments/temporal/ml-20m/train_from_scratch/dense_all_action_++  --model=dense_all_action_plus_plus --frozen_item=false --temporal=true --finetune=false --load_emb=false

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=100  --lr=0.001 --loss_function=sampled_softmax --log_dir=experiments/temporal/ml-20m/train_from_scratch/combined  --model=combined --frozen_item=false --temporal=true --finetune=false --load_emb=false