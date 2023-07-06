python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=1 --num_epochs=15  --lr=0.001 --loss_function=sampled_softmax --log_dir=test/ml-20m/finetune/normal_sasrec  --model=normal_sasrec --temporal=true --finetune=true --load_emb=true --frozen_item=false


python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=1 --num_epochs=15  --lr=0.001 --loss_function=sampled_softmax --log_dir=test/ml-20m/finetune/all_action_2  --model=all_action --temporal=true --finetune=true --load_emb=true --frozen_item=false

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=1 --num_epochs=15  --lr=0.001 --loss_function=sampled_softmax --log_dir=test/ml-20m/finetune/dense_all_action  --model=dense_all_action --temporal=true --load_emb=true --finetune=true --frozen_item=false

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=1 --num_epochs=15  --lr=0.001 --loss_function=sampled_softmax --log_dir=test/ml-20m/finetune/dense_all_action_+  --model=dense_all_action_plus --temporal=true --load_emb=true --finetune=true --frozen_item=false

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=1 --num_epochs=15  --lr=0.001 --loss_function=sampled_softmax --log_dir=test/ml-20m/finetune/dense_all_action_++  --model=dense_all_action_plus_plus --temporal=true --load_emb=true --finetune=true --frozen_item=false

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=1 --num_epochs=15  --lr=0.001 --loss_function=sampled_softmax --log_dir=test/ml-20m/finetune/combined  --model=integrated --temporal=true --load_emb=true --finetune=true --frozen_item=false