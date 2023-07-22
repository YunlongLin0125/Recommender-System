# python main.py --dataset=ml-20m --log_dir=test --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=200 --temporal=true --lr=0.003 --maxlen=200


python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_mod/finetune/all_action --device=cuda --model=all_action --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=20 --temporal=true --lr=0.001 --load_emb=true --frozen_item=false --finetune=true --maxlen=200

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_mod/finetune/combined --device=cuda --model=integrated --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=20 --temporal=true --lr=0.001 --load_emb=true --frozen_item=false --finetune=true --maxlen=200

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_mod/finetune/dense_all_action --device=cuda --model=dense_all_action --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=20 --temporal=true --lr=0.001 --load_emb=true --finetune=true --frozen_item=false --maxlen=200