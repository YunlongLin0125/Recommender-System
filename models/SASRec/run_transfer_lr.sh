#python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=20  --lr=0.003 --loss_function=sampled_softmax --log_dir=experiments/temporal/ml-20m/transfer_lr/lr=0.003/normal_sasrec  --model=normal_sasrec --frozen_item=true --temporal=true --finetune=false --load_emb=true

#python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=20  --lr=0.003 --loss_function=sampled_softmax --log_dir=experiments/temporal/ml-20m/transfer_lr/lr=0.003/all_action  --model=all_action --frozen_item=true --temporal=true --finetune=false --load_emb=true

#python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=20  --lr=0.003 --loss_function=sampled_softmax --log_dir=experiments/temporal/ml-20m/transfer_lr/lr=0.003/dense_all_action  --model=dense_all_action --frozen_item=true --temporal=true --finetune=false --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=20  --lr=0.003 --loss_function=sampled_softmax --log_dir=experiments/temporal/ml-20m/transfer_lr/lr=0.003/combined  --model=integrated --frozen_item=true --temporal=true --finetune=false --load_emb=true

#python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=20  --lr=0.001 --loss_function=sampled_softmax --log_dir=experiments/temporal/ml-20m/transfer_lr/lr=0.003/dense_all_action_+  --model=dense_all_action_plus --frozen_item=true --temporal=true --finetune=false --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=2 --num_epochs=20  --lr=0.003 --loss_function=sampled_softmax --log_dir=experiments/temporal/ml-20m/transfer_lr/lr=0.003/dense_all_action_++  --model=dense_all_action_plus_plus --frozen_item=true --temporal=true --finetune=false --load_emb=true

