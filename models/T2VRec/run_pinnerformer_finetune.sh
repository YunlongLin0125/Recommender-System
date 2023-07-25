python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_earlystop/finetune/all_action --device=cuda --model=all_action --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=100 --temporal=true --lr=0.001 --load_emb=true --frozen_item=false --maxlen=200 --finetune=true

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_earlystop/finetune/dense_all_action --device=cuda --model=dense_all_action --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=100 --temporal=true --lr=0.001 --load_emb=true --frozen_item=false --maxlen=200 --finetune=true

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_earlystop/finetune/combined --device=cuda --model=integrated --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=100 --temporal=true --lr=0.001 --load_emb=true --frozen_item=false --maxlen=200 --finetune=true

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_earlystop/finetune/dense_all_action_+ --device=cuda --model=dense_all_action_plus --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=100 --temporal=true --lr=0.001 --load_emb=true --frozen_item=false --maxlen=200 --finetune=true

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_earlystop/finetune/dense_all_action_++ --device=cuda --model=dense_all_action_plus_plus --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=100 --temporal=true --lr=0.001 --load_emb=true --frozen_item=false --maxlen=200 --finetune=true

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_earlystop/finetune/normal_sasrec --device=cuda --model=normal_sasrec --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=100 --temporal=true --lr=0.001 --load_emb=true --frozen_item=false --maxlen=200 --finetune=true