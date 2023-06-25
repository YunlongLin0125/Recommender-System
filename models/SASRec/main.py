import os
import time
import torch
import argparse

from model import SASRec, AllAction, DenseAll, SASRecSampledLoss, AllActionSampledLoss
from utils import *
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
start_time = time.time()

# Define global variables
DENSE_ALL_ACTION = 'dense_all_action'
ALL_ACTION = 'all_action'
NORMAL_SASREC = 'normal_sasrec'
SASREC_SAMPLED = 'sasrec_sampled'

BCE = 'bce'
SAMPLED_SOFTMAX = 'sampled_softmax'

# Create a list of all model names
MODEL_NAMES = [DENSE_ALL_ACTION, ALL_ACTION, NORMAL_SASREC, SASREC_SAMPLED]
LOSS_FUNCTIONS = [BCE, SAMPLED_SOFTMAX]


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--model', default=DENSE_ALL_ACTION, choices=MODEL_NAMES, required=True)
parser.add_argument('--loss_function', default=BCE, choices=LOSS_FUNCTIONS, required=True)
parser.add_argument('--dataset', required=True)
parser.add_argument('--log_dir', required=True)  # train_dir
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--window_size', default=7, type=int)
parser.add_argument('--window_predictor', default=False, type=str2bool)
parser.add_argument('--sas_window_eval', default=False, type=str2bool)
parser.add_argument('--window_eval', default=True, type=str2bool)
parser.add_argument('--eval_epoch', default=20, type=int)

args = parser.parse_args()
# dataset = data_partition(args.dataset)data
# args.dataset = 'reproduce/' + args.dataset
if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)
with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    if args.model == NORMAL_SASREC:  # normal sasrec but trained on the window train sequence
        dataset_window = data_partition_window_P(args.dataset, valid_percent=0.1, test_percent=0.1, train_percent=0.1)
        [_, user_train, user_valid, user_test, usernum, itemnum, _] = dataset_window
        sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen,
                              n_workers=3)
        dataset = dataset_window
        sample_train = user_train
        sample_num = usernum

    elif args.model == SASREC_SAMPLED:  # sasrec changes the data split to window structure
        dataset_window = data_partition_window_P(args.dataset, valid_percent=0.1, test_percent=0.1, train_percent=0.1)
        [sample_train, user_train, user_valid, user_test, usernum, itemnum, sample_num] = dataset_window
        sampler = WarpSampler(sample_train, sample_num, itemnum, batch_size=args.batch_size, maxlen=args.maxlen,
                              n_workers=3)
        dataset = dataset_window
        sample_train = sample_train
        sample_num = sample_num

    elif args.model == ALL_ACTION:  # all action prediction
        dataset_all_action = data_partition_window_dense_all_P(args.dataset, valid_percent=0.1,
                                                               test_percent=0.1, train_percent=0.1)
        [user_input, user_target, user_train, user_valid, user_test, usernum, itemnum] = dataset_all_action
        sampler = WarpSamplerAllAction(user_input, user_target, usernum, itemnum, batch_size=args.batch_size,
                                       maxlen=args.maxlen, n_workers=3)
        dataset = dataset_all_action
        sample_train = user_input
        sample_num = usernum

    elif args.model == DENSE_ALL_ACTION:  # dense all action prediction
        dataset_dense_all = data_partition_window_dense_all_P(args.dataset, valid_percent=0.1,
                                                              test_percent=0.1, train_percent=0.1)
        [user_input, user_target, user_train, user_valid, user_test, usernum, itemnum] = dataset_dense_all
        sampler = WarpSamplerDenseAll(user_input, user_target, usernum, itemnum, batch_size=args.batch_size,
                                      maxlen=args.maxlen, n_workers=3)
        dataset = dataset_dense_all
        sample_train = user_input
        sample_num = usernum

    else:  # Next item prediction SASRec
        dataset = data_partition(args.dataset)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset
        sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen,
                              n_workers=3)
        dataset = dataset
        sample_train = user_train
        sample_num = usernum

    f = open(os.path.join(args.log_dir, 'log.txt'), 'w')
    num_batch = len(sample_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in sample_train:
        cc += len(sample_train[u])
    print('average sequence length: %.2f' % (cc / len(sample_train)))
    print('number of training data: %.2f' % len(sample_train))
    print('number of items: %.2f' % itemnum)

    if args.model == ALL_ACTION:
        if args.loss_function == BCE:  # binary cross entropy loss
            model = AllAction(usernum, itemnum, args).to(args.device)
        else:
            model = AllActionSampledLoss(usernum, itemnum, args).to(args.device)

    elif args.model == DENSE_ALL_ACTION:
        if args.loss_function == BCE:  # binary cross entropy loss
            model = DenseAll(usernum, itemnum, args).to(args.device)
        else:
            model = DenseAll(usernum, itemnum, args).to(args.device)

    else:
        if args.loss_function == BCE:  # binary cross entropy loss
            model = SASRec(usernum, itemnum, args).to(args.device)
            # no ReLU activation in original SASRec implementation?
        else:  # args.loss_function is Sampled softmax loss
            model = SASRecSampledLoss(usernum, itemnum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train()  # enable model training
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb

            pdb.set_trace()

    if args.inference_only:
        if not args.window_eval:
            model.eval()
            t_test = evaluate(model, dataset, args)
            print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
        else:
            # t_test = evaluate_window_test(model, dataset, args)
            t_test = evaluate_window(model, dataset, args, eval_type='test')
            print('test (R@10: %.4f, P90coverage@10: %.4f)' % (t_test[0], t_test[1]))
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    # going to change the loss function here.
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break  # just to decrease identition
        # BCE loss
        if args.loss_function == BCE:
            for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = model(u, seq, pos, neg)
                logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)], dim=-1)
                # pos_logits.shape = (batch_size, num_targets)
                # neg_logits.shape = (batch_size, num_negs)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                       device=args.device)
                labels = torch.cat([pos_labels.unsqueeze(-1), neg_labels.unsqueeze(-1)], dim=-1)
                # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits)
                # check pos_logits > 0, neg_logits < 0
                adam_optimizer.zero_grad()
                # In case of multiple negative samples, use view to match dimensions
                # normal training
                if args.model == ALL_ACTION:
                    # all action train
                    # pos_indices = np.where(pos != 0)
                    loss = bce_criterion(pos_logits, pos_labels)
                    loss += bce_criterion(neg_logits, neg_labels)
                else:
                    # dense all action train and normal train (same)
                    indices = np.where(pos != 0)
                    # select from no padding
                    loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                    loss += bce_criterion(neg_logits[indices], neg_labels[indices])

                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step,
                                                                 loss.item()))  # expected 0.4~0.6 after init few epochs
        elif args.loss_function == SAMPLED_SOFTMAX:
            # sampled softmax loss
            for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = model(u, seq, pos, neg)
                # if model == normal_sasrec
                # pos_logits.shape = [batch_size, sequence_length]
                # neg_logits.shape = [batch_size, sequence_length, num_negs]
                # if model == all_action
                # pos_logits.shape = [batch_size, num_targets]
                # neg_logits.shape = [batch_size, num_negs]
                # if model == dense_all_action
                # pos_logits.shape = [batch_size, sequence_length]
                # neg_logits.shape = [batch_size, sequence_length, num_negs]
                if args.model == ALL_ACTION:
                    softmax_denominator = torch.sum(torch.exp(neg_logits), dim=-1).unsqueeze(-1) + torch.exp(pos_logits)
                    # softmax_denominator.shape = [batch_size, num_targets]
                    loss = - torch.log(torch.exp(pos_logits) / softmax_denominator)
                    # loss = -pos_logits + softmax_denominator
                else:
                    logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
                    # logits.shape = [batch_size, sequence_length, 1 + num_negs]
                    softmax_denominator = torch.logsumexp(logits, dim=-1)
                    loss = -pos_logits + softmax_denominator
                adam_optimizer.zero_grad()
                # In case of multiple negative samples, use view to match dimensions
                # normal training
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
                average_loss = loss.mean()
                average_loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step,
                                                                 average_loss.item()))
        if epoch % args.eval_epoch == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            if not args.window_eval:
                t_valid = evaluate_valid(model, dataset, args)
                t_test = evaluate(model, dataset, args)
                print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                      % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            else:
                # t_valid = evaluate_window_valid(model, dataset, args)
                # t_test = evaluate_window_test(model, dataset, args)
                t_valid = evaluate_window(model, dataset, args, eval_type='valid')
                t_test = evaluate_window(model, dataset, args, eval_type='test')
                print('epoch:%d, time: %f(s), valid (R@10: %.4f, nn: %.4f), test (R@10: %.4f, nn: %.4f)'
                      % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
                # print('epoch:%d, time: %f(s), valid (R@10: %.4f, P90coverage@10: %.4f), test (R@10: %.4f, '
                #       'P90coverage@10: %.4f)'
                #       % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            # folder = args.dataset + '_' + args.log_dir
            folder = args.log_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                                 args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
