python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_rel/frozen_item/all_action --device=cuda --model=all_action --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=20 --temporal=true --lr=0.003 --load_emb=true --frozen_item=true --maxlen=200

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_rel/frozen_item/dense_all_action --device=cuda --model=dense_all_action --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=20 --temporal=true --lr=0.003 --load_emb=true --frozen_item=true --maxlen=200

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_rel/frozen_item/combined --device=cuda --model=integrated --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=20 --temporal=true --lr=0.003 --load_emb=true --frozen_item=true --maxlen=200

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_rel/frozen_item/dense_all_action_+ --device=cuda --model=dense_all_action_plus --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=20 --temporal=true --lr=0.003 --load_emb=true --frozen_item=true --maxlen=200

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_rel/frozen_item/dense_all_action_++ --device=cuda --model=dense_all_action_plus_plus --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=20 --temporal=true --lr=0.003 --load_emb=true --frozen_item=true --maxlen=200

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer_rel/frozen_item/normal_sasrec --device=cuda --model=normal_sasrec --loss_function=sampled_softmax --eval_epoch=2 --num_epochs=20 --temporal=true --lr=0.003 --load_emb=true --frozen_item=true --maxlen=200

python main.py --dataset=ml-20m --log_dir=experiments/ml-20m/pinnerformer/train_from_scratch/normal_sasrec --device=cuda --model=normal_sasrec --loss_function=sampled_softmax --eval_epoch=50 --num_epochs=200 --temporal=true --lr=0.005 --maxlen=200