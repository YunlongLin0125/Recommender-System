cd ..
# ------------------------------------------------C_EXP1
'''
python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=30 --num_epochs=30  --lr=0.001 --loss_function=sampled_softmax --log_dir=timerecord_20ep/Scratch/normal_sasrec/1 --temporal=true --model=normal_sasrec --k_fold=1

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=30 --num_epochs=30  --lr=0.001 --loss_function=sampled_softmax --log_dir=timerecord_20ep/Scratch/all_action/1 --temporal=true --model=all_action --k_fold=1

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=30 --num_epochs=30  --lr=0.001 --loss_function=sampled_softmax --log_dir=timerecord_20ep/Scratch/dense_all_action/1 --temporal=true --model=dense_all_action --k_fold=1

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=30 --num_epochs=30  --lr=0.001 --loss_function=sampled_softmax --log_dir=timerecord_20ep/Scratch/integrated/1 --temporal=true --model=integrated --k_fold=1

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=30 --num_epochs=30  --lr=0.001 --loss_function=sampled_softmax --log_dir=timerecord_20ep/Scratch/dense_all_action_plus/1 --temporal=true --model=dense_all_action_plus --k_fold=1

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=30 --num_epochs=30  --lr=0.001 --loss_function=sampled_softmax --log_dir=timerecord_20ep/Scratch/dense_all_action_plus_plus/1 --temporal=true --model=dense_all_action_plus_plus --k_fold=1

# ------------------------------------------------C_EXP2
python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=30 --num_epochs=30  --lr=0.001 --loss_function=sampled_softmax --log_dir=timerecord_20ep/transfer/normal_sasrec/1 --temporal=true --model=normal_sasrec --k_fold=1 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=30 --num_epochs=30  --lr=0.001 --loss_function=sampled_softmax --log_dir=timerecord_20ep/transfer/all_action/1 --temporal=true --model=all_action --k_fold=1 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=30 --num_epochs=30  --lr=0.001 --loss_function=sampled_softmax --log_dir=timerecord_20ep/transfer/dense_all_action/1 --temporal=true --model=dense_all_action --k_fold=1 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=30 --num_epochs=30  --lr=0.001 --loss_function=sampled_softmax --log_dir=timerecord_20ep/transfer/integrated/1 --temporal=true --model=integrated --k_fold=1 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=30 --num_epochs=30  --lr=0.001 --loss_function=sampled_softmax --log_dir=timerecord_20ep/transfer/dense_all_action_plus/1 --temporal=true --model=dense_all_action_plus --k_fold=1 --frozen_item=true --load_emb=true

python main.py --dataset=processed/ml-20m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=30 --num_epochs=30  --lr=0.001 --loss_function=sampled_softmax --log_dir=timerecord_20ep/transfer/dense_all_action_plus_plus/1 --temporal=true --model=dense_all_action_plus_plus --k_fold=1 --frozen_item=true --load_emb=true

# ------------------------------------------------D_EXP1

cd ../T2VRec

python main.py --dataset=ml-20m --log_dir=timerecord_20ep/Scratch/t2v_normal_sasrec/1 --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=30 --num_epochs=30 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1

python main.py --dataset=ml-20m --log_dir=timerecord_20ep/Scratch/t2v_all_action/1 --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=30 --num_epochs=30 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1

python main.py --dataset=ml-20m --log_dir=timerecord_20ep/Scratch/t2v_dense_all_action/1 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=30 --num_epochs=30 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1

python main.py --dataset=ml-20m --log_dir=timerecord_20ep/Scratch/t2v_integrated/1 --model=integrated --device=cuda --loss_function=sampled_softmax --eval_epoch=30 --num_epochs=30 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1

python main.py --dataset=ml-20m --log_dir=timerecord_20ep/Scratch/t2v_dense_all+/1 --model=dense_all_action_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=30 --num_epochs=30 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1

python main.py --dataset=ml-20m --log_dir=timerecord_20ep/Scratch/t2v_dense_all++/1 --model=dense_all_action_plus_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=30 --num_epochs=30 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1
'''
# ------------------------------------------------D_EXP2

#python main.py --dataset=ml-20m --log_dir=timerecord_20ep/transfer/t2v_normal_sasrec/1 --model=normal_sasrec --device=cuda --loss_function=sampled_softmax --eval_epoch=30 --num_epochs=30 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1 --load_emb=true --frozen_item=true

#python main.py --dataset=ml-20m --log_dir=timerecord_20ep/transfer/t2v_all_action/1 --model=all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=30 --num_epochs=30 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1 --load_emb=true --frozen_item=true

#python main.py --dataset=ml-20m --log_dir=timerecord_20ep/transfer/t2v_dense_all_action/1 --model=dense_all_action --device=cuda --loss_function=sampled_softmax --eval_epoch=30 --num_epochs=30 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1 --load_emb=true --frozen_item=true

#python main.py --dataset=ml-20m --log_dir=timerecord_20ep/transfer/t2v_integrated/1 --model=integrated --device=cuda --loss_function=sampled_softmax --eval_epoch=30 --num_epochs=30 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1 --load_emb=true --frozen_item=true

#python main.py --dataset=ml-20m --log_dir=timerecord_20ep/transfer/t2v_dense_all+/1 --model=dense_all_action_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=30 --num_epochs=30 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1 --load_emb=true --frozen_item=true

#python main.py --dataset=ml-20m --log_dir=timerecord_20ep/transfer/t2v_dense_all++/1 --model=dense_all_action_plus_plus --device=cuda --loss_function=sampled_softmax --eval_epoch=30 --num_epochs=30 --temporal=true --lr=0.001 --maxlen=200 --k_fold=1 --load_emb=true --frozen_item=true


