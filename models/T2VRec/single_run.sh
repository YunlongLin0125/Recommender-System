# python main.py --dataset=ml-20m --log_dir=test --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=10 --num_epochs=200 --temporal=true --lr=0.003 --maxlen=200


python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_mod/frozen_item_lr0.002/dense_all_action --device=cuda --model=dense_all_action --loss_function=sampled_softmax --eval_epoch=3 --num_epochs=30 --temporal=true --lr=0.002 --load_emb=true --frozen_item=true --maxlen=200

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_mod/frozen_item_lr0.002/all_action --device=cuda --model=all_action --loss_function=sampled_softmax --eval_epoch=3 --num_epochs=30 --temporal=true --lr=0.002 --load_emb=true --frozen_item=true --maxlen=200


python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_mod/frozen_item_lr0.002/integrated --device=cuda --model=integrated --loss_function=sampled_softmax --eval_epoch=3 --num_epochs=30 --temporal=true --lr=0.002 --load_emb=true --frozen_item=true --maxlen=200