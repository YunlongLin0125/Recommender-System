import os
import time
import torch
import argparse

from model import SASRec
from utils import *
import time
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
start_time = time.time()


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
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
parser.add_argument('--window_eval', default=False, type=str2bool)
parser.add_argument('--eval_epoch', default=20, type=int)
parser.add_argument('--dense_all_action', default=False, type=str2bool)
parser.add_argument('--all_action', default=False, type=str2bool)

args = parser.parse_args()
# dataset = data_partition(args.dataset)data
# args.dataset = 'reproduce/' + args.dataset
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args.dataset)
    # dataset_window = data_partition_window_fixed(args.dataset, valid_percent=0.1, test_percent=0.1, train_k=7)
    dataset_window = data_partition_window_P(args.dataset, valid_percent=0.1, test_percent=0.1, train_percent=0.1)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    sample_num = usernum
    sample_train = user_train
    if args.window_predictor:
        [sample_train, user_train, user_valid, user_test, usernum, itemnum, sample_num] = dataset_window
    if args.sas_window_eval:
        [_, user_train, user_valid, user_test, usernum, itemnum, _] = dataset_window
        sample_train = user_train
        sample_num = usernum
    # if args.dense_all_action:
    #     dataset_dense_all = data_partition_window_dense_all_P_changeSampling(args.dataset,
    #                                                                          valid_percent=0.1,
    #                                                                          test_percent=0.1,
    #                                                                          train_percent=0.1)
    #     [sample_train, user_train, user_valid, user_test, usernum, itemnum, samplenum] = dataset_dense_all
    num_batch = len(sample_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in sample_train:
        cc += len(sample_train[u])
    print('average sequence length: %.2f' % (cc / len(sample_train)))
    print('number of training data: %.2f' % len(sample_train))
    print('number of items: %.2f' % itemnum)
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    sampler = WarpSampler(sample_train, sample_num, itemnum, batch_size=args.batch_size, maxlen=args.maxlen,
                          n_workers=3)
    if args.dense_all_action:
        dataset_dense_all = data_partition_window_dense_all_P(args.dataset, valid_percent=0.1,
                                                              test_percent=0.1, train_percent=0.1)
        [user_input, user_target, user_train, user_valid, user_test, usernum, itemnum] = dataset_dense_all
        sampler = WarpSamplerDenseAll(user_input, user_target, usernum, itemnum, batch_size=args.batch_size,
                                      maxlen=args.maxlen, n_workers=3)

    if args.all_action:
        dataset_all_action = data_partition_window_dense_all_P(args.dataset, valid_percent=0.1,
                                                               test_percent=0.1, train_percent=0.1)
        [user_input, user_target, user_train, user_valid, user_test, usernum, itemnum] = dataset_all_action
        sampler = WarpSamplerAllAction(user_input, user_target, usernum, itemnum, batch_size=args.batch_size,
                                       maxlen=args.maxlen, n_workers=3)

    model = SASRec(usernum, itemnum, args).to(args.device)  # no ReLU activation in original SASRec implementation?

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
            t_valid = evaluate_window_valid(model, dataset_window, args)
            t_test = evaluate_window_test(model, dataset_window, args)
            print('test (R@10: %.4f, P90coverage@10: %.4f)' % (t_test[0], t_test[1]))
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break  # just to decrease identition
        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            # pos_logits.shape = (batch_size, num_targets)
            # neg_logits.shape = (batch_size, num_negs)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.device)

            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits)
            # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            # In case of multiple negative samples, use view to match dimensions
            # normal training

            if args.all_action:
                # all action train
                # pos_indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits, pos_labels)
                loss += bce_criterion(neg_logits, neg_labels)
            else:
                # dense all action train
                indices = np.where(pos != 0)
                # select from no padding
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step,
                                                             loss.item()))  # expected 0.4~0.6 after init few epochs

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
                if args.dense_all_action:
                    t_valid = evaluate_window_valid(model, dataset_dense_all, args)
                    t_test = evaluate_window_test(model, dataset_dense_all, args)
                elif args.all_action:
                    t_valid = evaluate_window_valid(model, dataset_all_action, args)
                    t_test = evaluate_window_test(model, dataset_all_action, args)
                else:
                    t_valid = evaluate_window_valid(model, dataset_window, args)
                    t_test = evaluate_window_test(model, dataset_window, args)
                # print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                #       % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
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
            folder = args.dataset + '_' + args.train_dir
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
