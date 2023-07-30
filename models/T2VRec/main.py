import os
import time
import torch
import pickle
import argparse
import sys
from model_sasrec import SASRecSampledLoss, SASRec
# from memory_profiler import profile
from model import T2V_SASRec, T2V_AllAction, T2V_DenseAllAction, T2V_DenseAllPlus
from model import T2V_SASRecSampledLoss, T2V_AllActionSampledLoss
from model import T2V_DenseAllActionSampledLoss, T2V_DenseAllPlusSampledLoss
from utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
start_time = time.time()

# Define global variables
DENSE_ALL_ACTION = 'dense_all_action'
ALL_ACTION = 'all_action'
NORMAL_SASREC = 'normal_sasrec'
SASREC_SAMPLED = 'sasrec_sampled'
DENSE_ALL_PLUS = 'dense_all_action_plus'
DENSE_ALL_PLUS_PLUS = 'dense_all_action_plus_plus'
INTEGRATED = 'integrated'

BCE = 'bce'
SAMPLED_SOFTMAX = 'sampled_softmax'

# Create a list of all model names
MODEL_NAMES = [DENSE_ALL_ACTION, ALL_ACTION, NORMAL_SASREC, SASREC_SAMPLED,
               DENSE_ALL_PLUS, INTEGRATED, DENSE_ALL_PLUS_PLUS]
LOSS_FUNCTIONS = [BCE, SAMPLED_SOFTMAX]


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--log_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.00005, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--eval_epoch', default=20, type=int)
parser.add_argument('--temporal', default=False, type=str2bool)
parser.add_argument('--model', choices=MODEL_NAMES, required=True)
parser.add_argument('--loss_function', default=BCE, choices=LOSS_FUNCTIONS, required=True)
parser.add_argument('--load_emb', default=False, type=str2bool)
parser.add_argument('--frozen_item', default=False, type=str2bool)
parser.add_argument('--finetune', default=False, type=str2bool)
parser.add_argument('--val_loss', default=False, type=str2bool)
parser.add_argument('--k_fold', default=0, type=int)

args = parser.parse_args()
if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)
with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()


# @profile
def run():
    # dataset split
    # if args.percentage:
    #     dataset = data_partition_window_InputTarget_byP(args.dataset, valid_percent=0.1, test_percent=0.1,
    #                                                     train_percent=0.1)
    #     [user_input, user_target, user_train, user_valid, user_test, usernum, itemnum, timenum] = dataset
    # temporal
    best_score = 0
    max_patience = 3
    patience = 0
    print("EXP: " + args.log_dir)
    if args.temporal:
        dataset = data_partition_window_InputTarget_byT(args.dataset + '_train_withtime',
                                                        args.dataset + '_target_withtime', args)
        [user_input, user_target, usernum, itemnum, train_users, valid_users, test_users] = dataset
        sample_train = user_input
    else:
        dataset = data_partition(args.dataset)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset
        sample_train = user_train
    # Model Selection
    if args.loss_function == BCE:
        if args.model == NORMAL_SASREC:
            # 1, 1
            sample_train = user_input
            model = T2V_SASRec(usernum, itemnum, args).to(args.device)
        else:
            # window_predictor
            sample_train = user_input
            if args.model == ALL_ACTION:
                # final embedding focus
                # num_pos, num_neg
                model = T2V_AllAction(usernum, itemnum, args).to(args.device)
            elif args.model == DENSE_ALL_ACTION:
                # 1, num neg
                model = T2V_DenseAllAction(usernum, itemnum, args).to(args.device)
            elif args.model in [DENSE_ALL_PLUS, DENSE_ALL_PLUS_PLUS, INTEGRATED]:
                # num_pos, num neg
                model = T2V_DenseAllPlus(usernum, itemnum, args).to(args.device)
            else:
                model = None
                quit()
    else:
        if args.model == NORMAL_SASREC:
            # 1, 1
            sample_train = user_input
            model = T2V_SASRecSampledLoss(usernum, itemnum, args).to(args.device)
        else:
            sample_train = user_input
            if args.model == ALL_ACTION:
                # final embedding focus
                # num_pos, num_neg
                model = T2V_AllActionSampledLoss(usernum, itemnum, args).to(args.device)
            elif args.model == DENSE_ALL_ACTION:
                # 1, num neg
                model = T2V_DenseAllActionSampledLoss(usernum, itemnum, args).to(args.device)
            elif args.model in [DENSE_ALL_PLUS, DENSE_ALL_PLUS_PLUS, INTEGRATED]:
                # num_pos, num neg
                model = T2V_DenseAllPlusSampledLoss(usernum, itemnum, args).to(args.device)
            else:
                model = None
                quit()
    print("Model : ", args.model)
    print("Loss : ", args.loss_function)

    num_batch = len(sample_train) // args.batch_size
    num_batch_valid = len(valid_users) // args.batch_size
    # usernum // args.batch_size
    cc = 0.0
    for u in sample_train:
        cc += len(sample_train[u])
    print('average sequence length: %.2f' % (cc / len(sample_train)))
    print('number of unique users: %.2f' % usernum)
    print('number of unique items: %.2f' % itemnum)

    f = open(os.path.join(args.log_dir, 'log.txt'), 'w')
    # Sampler Selection
    if args.temporal:
        # sampler used for temporal splitting data
        sampler = WarpSamplerInputTarget_byT(user_input, user_target, train_users, usernum, itemnum,
                                             args, batch_size=args.batch_size,
                                             maxlen=args.maxlen, n_workers=3)
    else:
        # normal sampler
        sampler = WarpSampler(sample_train, usernum, itemnum, batch_size=args.batch_size,
                              maxlen=args.maxlen, n_workers=3)

    # else:
    #     if args.model in [NORMAL_SASREC]:
    #         sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size,
    #                               maxlen=args.maxlen,
    #                               n_workers=3)
    #     else:
    #         sampler = WarpSamplerInputTarget(user_input, user_target, usernum, itemnum, args,
    #                                          batch_size=args.batch_size,
    #                                          maxlen=args.maxlen, n_workers=3)
    print("Sampler Generated Done")
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass  # just ignore those failed init layers

    # load item embedding
    if args.load_emb:
        if 'ml-20m' in args.dataset:
            path = 'F_experiments/T/ml-20m/transfer/item_emb/normal_sasrec.best.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth'
        item_emb_param = torch.load(path, map_location=args.device)['item_emb.weight']
        model.item_emb.weight.data = item_emb_param.clone()
        del item_emb_param  # Free the memory
        torch.cuda.empty_cache()  # Clear GPU cache
        print("Load item embedding")

    # freeze item embedding
    if args.frozen_item:
        for param in model.item_emb.parameters():
            param.requires_grad = False
        print("Frozen item embedding")

    return

    model.train()  # enable model training
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        print("Loading model")
        state_path = 'experiments/ml-20m/pinnerformer_earlystop/frozen_item/' \
                     'all_action/all_action.best.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth'
        model.load_state_dict(torch.load(state_path, map_location=torch.device(args.device)))
        print("Model Loaded")

    bce_criterion = torch.nn.BCEWithLogitsLoss()

    if args.inference_only:
        model.eval()
        if args.temporal:
            t_test = evaluate_T_withP90(model, dataset, args, 'test')
            # t_valid = evaluate_T(model, dataset, args, 'valid')
        else:
            t_test = evaluate_T_withP90(model, dataset, args, 'test')
            # t_valid = evaluate(model, dataset, args, 'valid')
        print('test (Recall@10: %.4f, P90: %.4f)' % (t_test[0], t_test[1]))
        f.write(str(t_test) + '\n')

    # configure finetune
    if args.finetune:
        item_emb_lr = args.lr * 1e-2
        # Separate the item embeddings parameters
        item_emb_parameters = model.item_emb.parameters()
        # Get all other parameters of the model
        other_parameters = [param for name, param in model.named_parameters() if 'item_emb' not in name]

        adam_optimizer = torch.optim.Adam([
            {'params': item_emb_parameters, 'lr': item_emb_lr},
            {'params': other_parameters, 'lr': args.lr}
        ], betas=(0.9, 0.98))
        print("Finetune item embedding")
    else:
        adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    print("Training Started")
    T = 0.0
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break  # just to decrease identition
        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, time_seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            # print(time_seq)
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            time_seq = np.array(time_seq)
            pos_logits, neg_logits = model(u, seq, time_seq, pos, neg)
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            # Loss function selection
            if args.loss_function == BCE:
                # bce loss
                if args.model == ALL_ACTION:
                    loss = bce_criterion(pos_logits, pos_labels)
                    loss += bce_criterion(neg_logits, neg_labels)
                else:
                    indices = (seq != 0)
                    loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                    loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            else:
                # sampled softmax loss
                if args.model not in [NORMAL_SASREC, SASREC_SAMPLED, DENSE_ALL_ACTION]:
                    if args.model != ALL_ACTION:
                        mask = (seq != 0)
                        pos_logits = pos_logits[mask]
                        neg_logits = neg_logits[mask]
                    softmax_denominator = \
                        torch.sum(torch.exp(neg_logits), dim=-1).unsqueeze(-1) + torch.exp(pos_logits)
                    softmax_denominator = torch.log(softmax_denominator)
                    loss = -pos_logits + softmax_denominator
                else:
                    # num_pos = 1, num_negs >= 1
                    mask = (seq != 0)
                    pos_logits = pos_logits[mask]
                    neg_logits = neg_logits[mask]
                    softmax_denominator = torch.sum(torch.exp(neg_logits), dim=-1) + torch.exp(pos_logits)
                    softmax_denominator = torch.log(softmax_denominator)
                    loss = -pos_logits + softmax_denominator
            # L2 regularization
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            if args.loss_function == 'sampled_softmax':
                loss = loss.mean()
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step,
                                                             loss.item()))

        if epoch % args.eval_epoch == 0:
            # modify the progress to validation loss
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            # if args.val_loss:
            #     valid_loss = Loss_calculation(user_input, user_target, valid_users, model, args)
            #     print('epoch:%d, time: %f(s), valid_loss : (%.4f)', epoch, T, valid_loss)
            #     f.write("Validation Loss in epoch " + str(epoch) + ' : ' + str(valid_loss / len(valid_users)) + '\n')
            # else:
            if args.temporal:
                # t_test = evaluate_T(model, dataset, args, 'test')
                t_valid = evaluate_T(model, dataset, args, 'valid')
            else:
                # t_test = evaluate(model, dataset, args, 'test')
                t_valid = evaluate(model, dataset, args, 'valid')
            print('epoch:%d, time: %f(s), valid (R@10: %.4f, P90: %.4f)'
                  % (epoch, T, t_valid[0], t_valid[1]))
            f.write("--time: " + str(T) + ", " + "epoch: " + str(epoch) + ", " + "Score: " + str(t_valid) + '\n')
            f.flush()

            if t_valid[0] > best_score:
                best_score = t_valid[0]
                folder = args.log_dir
                fname = '{}.best.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(args.model, args.lr, args.num_blocks,
                                     args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))
                patience = 0
            else:
                patience += 1

            if patience >= max_patience:
                # early stop the model
                f.write('Early stopping due to lack of improvement in validation loss.' + '\n')
                f.flush()
                print('Early stopping due to lack of improvement in validation loss.')
                break
        t0 = time.time()
        model.train()
        # if epoch == args.num_epochs:
        #     folder = args.log_dir
        #     fname = '{}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        #     fname = fname.format(args.model, args.num_epochs, args.lr, args.num_blocks, args.num_heads,
        #                          args.hidden_units,
        #                          args.maxlen)
        #     torch.save(model.state_dict(), os.path.join(folder, fname))
    # Final model performance
    model.load_state_dict(torch.load(os.path.join(args.log_dir, fname)))
    t_test = evaluate_T_withP90(model, dataset, args, 'test')
    f.write("Final Model Performance : " + str(t_test) + '\n')

    end_time = time.time()
    execution_time = end_time - start_time
    f.write("Execution time: " + str(execution_time) + "seconds" + '\n')
    f.close()
    sampler.close()
    print("Done")


if __name__ == '__main__':
    run()
