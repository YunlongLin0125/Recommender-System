cd ..

#----------------------------------normal_sasrec
python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/normal_sasrec/1 --model=normal_sasrec --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/normal_sasrec/2 --model=normal_sasrec --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/normal_sasrec/3 --model=normal_sasrec --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/normal_sasrec/4 --model=normal_sasrec --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/normal_sasrec/5 --model=normal_sasrec --load_emb=true --frozen_item=true


#----------------------------------all_action
python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/all_action/1 --model=all_action --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/all_action/2 --model=all_action --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/all_action/3 --model=all_action --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/all_action/4 --model=all_action --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/all_action/5 --model=all_action --load_emb=true --frozen_item=true

#----------------------------------dense_all_action
python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action/1 --model=dense_all_action --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action/2 --model=dense_all_action --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action/3 --model=dense_all_action --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action/4 --model=dense_all_action --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action/5 --model=dense_all_action --load_emb=true --frozen_item=true


#----------------------------------integrated
python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/integrated/1 --model=integrated --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/integrated/2 --model=integrated --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/integrated/3 --model=integrated --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/integrated/4 --model=integrated --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/integrated/5 --model=integrated --load_emb=true --frozen_item=true


#----------------------------------dense_all_action_plus
python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus/1 --model=dense_all_action_plus --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus/2 --model=dense_all_action_plus --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus/3 --model=dense_all_action_plus --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus/4 --model=dense_all_action_plus --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus/5 --model=dense_all_action_plus --load_emb=true --frozen_item=true


#----------------------------------dense_all_action_plus_plus
python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus_plus/1 --model=dense_all_action_plus_plus --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus_plus/2 --model=dense_all_action_plus_plus --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus_plus/3 --model=dense_all_action_plus_plus --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus_plus/4 --model=dense_all_action_plus_plus --load_emb=true --frozen_item=true

python main.py --dataset=processed/ml-1m --maxlen=200 --dropout_rate=0.2 --device=cuda --window_eval=true --eval_epoch=5 --num_epochs=1000  --lr=0.001 --loss_function=sampled_softmax --log_dir=F_experiments/P/ml-1m/transfer/sampled_softmax/lr=0.001/dense_all_action_plus_plus/5 --model=dense_all_action_plus_plus --load_emb=true --frozen_item=true