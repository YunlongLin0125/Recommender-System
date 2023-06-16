import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict, Counter
from multiprocessing import Process, Queue


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():
        # randomly sampled a valid user
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            # fill the seq with all interacted except the last item as the target action
            # i started by user_train[user][-2]
            seq[idx] = i  # [x,x,x,x,x,x,#]
            pos[idx] = nxt  # [#,y,y,y,y,y,y]
            # negative sampling (random_neq) from 1 to itemnum + 1
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        # user: uniformly sampled userid
        # seq: the sequence of items the user has interacted with
        # pos: positive samples,
        # neg: the negative samples respectively
        return user, seq, pos, neg

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            # batch
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def data_partition_window_forSAS(fname, valid_percent, test_percent):
    if valid_percent + test_percent > 0.6:
        print('the percent you select for val/test are too high')
        return None
    valid_start = 1 - valid_percent - test_percent
    test_start = 1 - test_percent
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train_seq = {}
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    # read from each line
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        # count user and items
    # read from each user
    count = 0
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:

            # select the whole training seq
            # user_train[user] = User[user][:-2]
            seq_len = len(User[user])
            valid_index = int(seq_len * valid_start)
            test_index = int(seq_len * test_start)
            if valid_index == test_index:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                train_seq = User[user][: valid_index]
                valid_seq = User[user][valid_index: test_index]
                test_seq = User[user][test_index:]
                user_train[user] = train_seq
                user_valid[user] = []
                user_valid[user] += valid_seq
                user_test[user] = []
                user_test[user] += test_seq
    return [user_train, user_valid, user_test, usernum, itemnum]


def data_partition_window_P(fname, valid_percent, test_percent, train_percent):
    if valid_percent + test_percent > 0.6:
        print('the percent you select for val/test are too high')
        return None
    valid_start = 1 - valid_percent - test_percent
    test_start = 1 - test_percent
    train_start = 1 - train_percent
    usernum = 0
    itemnum = 0
    samplenum = 0
    User = defaultdict(list)
    user_train_seq = {}
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    # read from each line
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        # count user and items
    # read from each user
    count = 0
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            # select the whole training seq
            # user_train[user] = User[user][:-2]
            seq_len = len(User[user])
            valid_index = int(seq_len * valid_start)
            test_index = int(seq_len * test_start)
            if valid_index == test_index:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                train_seq = User[user][: valid_index]
                valid_seq = User[user][valid_index: test_index]
                test_seq = User[user][test_index:]
                train_seq_length = len(train_seq)
                split_index = int(train_seq_length * train_start)
                input_seq = train_seq[:split_index]
                target_seq = train_seq[split_index:]
                for target in target_seq:
                    count += 1
                    user_train[count] = input_seq + [target]
                user_train_seq[user] = []
                user_train_seq[user] += train_seq
                user_valid[user] = []
                user_valid[user] += valid_seq
                user_test[user] = []
                user_test[user] += test_seq
    samplenum = count
    return [user_train, user_train_seq, user_valid, user_test, usernum, itemnum, samplenum]


def data_partition_window_fixed(fname, valid_percent, test_percent, train_k):
    if valid_percent + test_percent > 0.6:
        print('the percent you select for val/test are too high')
        return None
    valid_start = 1 - valid_percent - test_percent
    test_start = 1 - test_percent
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train_seq = {}
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    # read from each line
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        # count user and items
    # read from each user
    count = 0
    for user in User:
        seq_len = len(User[user])
        valid_index = int(seq_len * valid_start)
        target_index = valid_index - train_k
        test_index = int(seq_len * test_start)
        if valid_index == test_index:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        if target_index <= 1:
            continue
        else:
            train_seq = User[user][: target_index]
            train_target = User[user][target_index: valid_index]
            valid_seq = User[user][valid_index: test_index]
            test_seq = User[user][test_index:]
            for target in train_target:
                count += 1
                user_train[count] = train_seq + [target]
            user_train_seq[user] = []
            user_train_seq[user] += train_seq
            user_valid[user] = []
            user_valid[user] += valid_seq
            user_test[user] = []
            user_test[user] += test_seq
    samplenum = count
    return [user_train, user_train_seq, user_valid, user_test, usernum, itemnum, samplenum]


def data_partition_window(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    split_percent = 0.9
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    # read from each line
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    # read from each user
    count = 0
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            # select the whole training seq
            # user_train[user] = User[user][:-2]
            train_seq = User[user][:-2]
            train_seq_length = len(train_seq)
            split_index = int(train_seq_length * split_percent)
            input_seq = train_seq[:split_index]
            target_seq = train_seq[split_index:]
            for target in target_seq:
                count += 1
                user_train[count] = input_seq + [target]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    usernum = count
    return [user_train, user_valid, user_test, usernum, itemnum]


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]  # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    NDCG = 0.0
    valid_user = 0.0
    k = 7
    HT = 0.0
    item_set = set(range(1, itemnum + 1))
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < k: continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        # sampled softmax
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # full softmax
        # item_set = set(range(1, itemnum + 1))
        # item_set = item_set - rated
        # item_idx += list(item_set)
        # only sampling 100 instances
        # for _ in range(100):
        #     t = np.random.randint(1, itemnum + 1)
        #     while t in rated: t = np.random.randint(1, itemnum + 1)
        #     item_idx.append(t)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# fixed window size
# def evaluate_window_valid(model, dataset_window, args):
#     [_, train, valid, test, usernum, itemnum, samplenum] = copy.deepcopy(dataset_window)
#     NDCG = 0.0
#     valid_user = 0.0
#     HT = 0.0
#     k = 7
#     sample_nums = 100
#     if usernum > 10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
#     for u in users:
#         if len(train[u]) < 1 or len(valid[u]) < k: continue
#         # validation_num = len(valid[u])
#         validation_num = 7
#         nDCG_u = 0
#         ht_u = 0
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             # fill the sequence from end to beginning
#             idx -= 1
#             if idx == -1: break
#             # select the max len or all of the training data in the sequence
#             # limit the length, seq contains the actual training sequence
#         # interacted items
#         rated = set(train[u])
#         rated.add(0)
#         # process items
#         for valid_id in range(validation_num):
#             item_idx = [valid[u][valid_id]]
#             for _ in range(sample_nums):
#                 t = np.random.randint(1, itemnum + 1)
#                 while t in rated: t = np.random.randint(1, itemnum + 1)
#                 item_idx.append(t)
#             predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
#             predictions = predictions[0]
#             rank = predictions.argsort().argsort()[0].item()
#             if rank < 10:
#                 nDCG_u += 1 / np.log2(rank + 2)
#                 ht_u += 1
#         NDCG += nDCG_u / validation_num
#         HT += ht_u / validation_num
#         valid_user += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
#     print(valid_user)
#     return NDCG / valid_user, HT / valid_user
#
#
# def evaluate_window_test(model, dataset_window, args):
#     [_, train, valid, test, usernum, itemnum, samplenum] = copy.deepcopy(dataset_window)
#     NDCG = 0.0
#     valid_user = 0.0
#     HT = 0.0
#     k = 7
#     sample_nums = 100
#     if usernum > 10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
#     for u in users:
#         if len(train[u]) < 1 or len(test[u]) < k: continue
#         test_num = len(test[u])
#         test_num = 7
#         nDCG_u = 0
#         ht_u = 0
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i in reversed(train[u] + valid[u]):
#             seq[idx] = i
#             # fill the sequence from end to beginning
#             idx -= 1
#             if idx == -1: break
#             # select the max len or all of the training data in the sequence
#             # limit the length, seq contains the actual training sequence
#         # interacted items
#         rated = set(train[u] + valid[u])
#         rated.add(0)
#         # process items
#         for test_id in range(test_num):
#             item_idx = [test[u][test_id]]
#             for _ in range(sample_nums):
#                 t = np.random.randint(1, itemnum + 1)
#                 while t in rated: t = np.random.randint(1, itemnum + 1)
#                 item_idx.append(t)
#             predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
#             predictions = predictions[0]
#             rank = predictions.argsort().argsort()[0].item()
#             if rank < 10:
#                 nDCG_u += 1 / np.log2(rank + 2)
#                 ht_u += 1
#         NDCG += nDCG_u / test_num
#         HT += ht_u / test_num
#         valid_user += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
#     # print(valid_user)
#     return NDCG / valid_user, HT / valid_user

# (Hit Ratio, nDCG) on percentage
# def evaluate_window_valid(model, dataset_window, args):
#     [_, train, valid, test, usernum, itemnum, samplenum] = copy.deepcopy(dataset_window)
#     NDCG = 0.0
#     valid_user = 0.0
#     HT = 0.0
#     sample_nums = 100
#
#     if usernum > 10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
#     for u in users:
#         if len(train[u]) < 1 or len(valid[u]) < 1: continue
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             # fill the sequence from end to beginning
#             idx -= 1
#             if idx == -1: break
#             # select the max len or all of the training data in the sequence
#             # limit the length, seq contains the actual training sequence
#         # interacted items
#         rated = set(train[u])
#         rated.add(0)
#         # process items
#         item_idx = valid[u]
#         for _ in range(sample_nums):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)
#         valid_num = len(valid[u])
#         # collect all indexes, which needs to process on
#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])[0]
#         # ranks = predictions.argsort().argsort()[:valid_num].item()
#         # ranks = predictions.argsort().argsort()[:valid_num]
#         ranks = [i.item() for i in predictions.argsort().argsort()[:valid_num]]
#         valid_user += 1
#         # update the metric value
#         if min(ranks) < 10:
#             HT += 1
#         rankCount = 0
#         iDCG = 0
#         DCG = 0
#         for rank in ranks:
#             if rankCount < 10:
#                 iDCG += 1 / np.log2(rankCount + 2)
#                 rankCount += 1
#             if rank < 10:
#                 DCG += 1 / np.log2(rank + 2)
#         NDCG += DCG / iDCG
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
#     return NDCG / valid_user, HT / valid_user
#
#
# def evaluate_window_test(model, dataset_window, args):
#     [_, train, valid, test, usernum, itemnum, samplenum] = copy.deepcopy(dataset_window)
#     NDCG = 0.0
#     valid_user = 0.0
#     HT = 0.0
#     # P90 coverage means the smallest item sets that appear in the top 10 lists of at least 90% of the users.
#     sample_nums = 100
#     # random_items = random.sample(range(1, itemnum + 1), sample_nums)
#     # sample_idx = random_items
#     # sample_idx_tensor = torch.tensor(sample_idx).to(args.device)
#     if usernum > 10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
#     for u in users:
#         if len(train[u]) < 1 or len(test[u]) < 1: continue
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i in reversed(train[u] + valid[u]):
#             seq[idx] = i
#             # fill the sequence from end to beginning
#             idx -= 1
#             if idx == -1: break
#             # select the max len or all of the training data in the sequence
#             # limit the length, seq contains the actual training sequence
#         # interacted items
#         rated = set(train[u] + valid[u])
#         rated.add(0)
#         # process items
#         item_idx = test[u]
#         for _ in range(sample_nums):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)
#         valid_num = len(test[u])
#         # collect all indexes, which needs to process on
#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])[0]
#         # ranks = predictions.argsort().argsort()[:valid_num].item()
#         # ranks = predictions.argsort().argsort()[:valid_num]
#         ranks = [i.item() for i in predictions.argsort().argsort()[:valid_num]]
#         valid_user += 1
#         # update the metric value
#         if min(ranks) < 10:
#             HT += 1
#         rankCount = 0
#         iDCG = 0
#         DCG = 0
#         for rank in ranks:
#             rankCount += 1
#             if rankCount < 10:
#                 iDCG += 1 / np.log2(rankCount + 2)
#             if rank < 10:
#                 DCG += 1 / np.log2(rank + 2)
#         NDCG += DCG / iDCG
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
#     return NDCG / valid_user, HT / valid_user

## (Recall@k, P90 Coverage)
# def evaluate_window_valid(model, dataset, dataset_window, args):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
#     [_, train, valid, test, _, itemnum, sample_num] = copy.deepcopy(dataset_window)
#     Recall = 0.0
#     Recall_U = 0.0
#     coverage_list = []
#     # P90 coverage means the smallest item sets that appear in the top 10 lists of at least 90% of the users.
#     valid_user = 0.0
#     sample_nums = 500
#     random_items = random.sample(range(1, itemnum + 1), sample_nums)
#     sample_idx = random_items
#     sample_idx_tensor = torch.tensor(sample_idx).to(args.device)
#     users = range(1, usernum + 1)
#     for u in users:
#         if len(train[u]) < 1 or len(valid[u]) < 1: continue
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             # fill the sequence from end to beginning
#             idx -= 1
#             if idx == -1: break
#             # select the max len or all of the training data in the sequence
#             # limit the length, seq contains the actual training sequence
#         # interacted items
#         rated = set(train[u])
#         rated.add(0)
#         # ground truth item
#         ground_truth_idx = valid[u]
#         valid_num = len(valid[u])
#         # collect all indexes, which needs to process on
#         process_idx = ground_truth_idx + sample_idx
#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], process_idx]])[0]
#         # target distance
#         target_ds = predictions[:valid_num]
#         # sampled results
#         sample_d = predictions[valid_num:]
#         # print(len(sample_d))
#         for target_d in target_ds:
#             bool_tensor = target_d >= sample_d
#             count = torch.sum(bool_tensor).item()
#             if count < 10:
#                 Recall_U += 1
#         Recall_U = Recall_U / valid_num
#         Recall += Recall_U
#         Recall_U = 0
#         sorted_indices = torch.argsort(sample_d)
#         sorted_sample_idx = sample_idx_tensor[sorted_indices]
#         # take the coverage@10 for all users
#         coverage_list += list(sorted_sample_idx[:10])
#         valid_user += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
#     p90_list = [i.item() for i in coverage_list]
#     p90_dict = Counter(p90_list)
#     p90_sort = sorted(p90_dict.items(), key=lambda x: x[1], reverse=True)
#     total_rec = 0
#     item_count = 0
#     for _, num in p90_sort:
#         total_rec += num
#         item_count += 1
#         if total_rec >= 0.9 * 10 * usernum:
#             break
#     return Recall / valid_user, item_count / sample_nums
#
#
# def evaluate_window_test(model, dataset, dataset_window, args):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
#     [_, train, valid, test, _, itemnum, sample_num] = copy.deepcopy(dataset_window)
#     Recall = 0.0
#     Recall_U = 0.0
#     coverage_list = []
#     # P90 coverage means the smallest item sets that appear in the top 10 lists of at least 90% of the users.
#     valid_user = 0.0
#     sample_nums = 500
#     random_items = random.sample(range(1, itemnum + 1), sample_nums)
#     sample_idx = random_items
#     sample_idx_tensor = torch.tensor(sample_idx).to(args.device)
#     users = range(1, usernum + 1)
#     for u in users:
#         if len(train[u]) < 1 or len(test[u]) < 1: continue
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i in reversed(train[u] + valid[u]):
#             seq[idx] = i
#             # fill the sequence from end to beginning
#             idx -= 1
#             if idx == -1: break
#             # select the max len or all of the training data in the sequence
#             # limit the length, seq contains the actual training sequence
#         # interacted items
#         rated = set(train[u]+valid[u])
#         rated.add(0)
#         # ground truth item
#         ground_truth_idx = test[u]
#         test_num = len(test[u])
#         # collect all indexes, which needs to process on
#         process_idx = ground_truth_idx + sample_idx
#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], process_idx]])[0]
#         # target distance
#         target_ds = predictions[:test_num]
#         # sampled results
#         sample_d = predictions[test_num:]
#         # print(len(sample_d))
#         for target_d in target_ds:
#             bool_tensor = target_d >= sample_d
#             count = torch.sum(bool_tensor).item()
#             if count < 10:
#                 Recall_U += 1
#         Recall_U = Recall_U / test_num
#         Recall += Recall_U
#         Recall_U = 0
#         sorted_indices = torch.argsort(sample_d)
#         sorted_sample_idx = sample_idx_tensor[sorted_indices]
#         # take the coverage@10 for all users
#         coverage_list += list(sorted_sample_idx[:10])
#         valid_user += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
#     p90_list = [i.item() for i in coverage_list]
#     p90_dict = Counter(p90_list)
#     p90_sort = sorted(p90_dict.items(), key=lambda x: x[1], reverse=True)
#     total_rec = 0
#     item_count = 0
#     for _, num in p90_sort:
#         total_rec += num
#         item_count += 1
#         if total_rec >= 0.9 * 10 * usernum:
#             break
#     return Recall / valid_user, item_count / sample_nums

# Recall@k, XX
def evaluate_window_valid(model, dataset_window, args):
    [_, train, valid, test, usernum, itemnum, sample_num] = copy.deepcopy(dataset_window)
    Recall = 0.0
    Recall_U = 0.0
    valid_user = 0.0
    sample_nums = 500
    users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            # fill the sequence from end to beginning
            idx -= 1
            if idx == -1: break
            # select the max len or all of the training data in the sequence
            # limit the length, seq contains the actual training sequence
        # interacted items
        rated = set(train[u])
        rated.add(0)
        # ground truth item
        neg = []
        for _ in range(sample_nums):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            neg.append(t)

        valid_num = len(valid[u])
        item_idx = valid[u] + neg
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])[0]
        # target distance
        target_ds = predictions[:valid_num]
        # sampled results
        sample_d = predictions[valid_num:]
        for target_d in target_ds:
            bool_tensor = target_d >= sample_d
            count = torch.sum(bool_tensor).item()
            if count < 10:
                Recall_U += 1
        Recall_U = Recall_U / valid_num
        Recall += Recall_U
        Recall_U = 0
        # take the coverage@10 for all users
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    return Recall / valid_user, 0.66


def evaluate_window_test(model, dataset_window, args):
    [_, train, valid, test, usernum, itemnum, sample_num] = copy.deepcopy(dataset_window)
    Recall = 0.0
    Recall_U = 0.0
    valid_user = 0.0
    sample_nums = 500
    users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u] + valid[u]):
            seq[idx] = i
            # fill the sequence from end to beginning
            idx -= 1
            if idx == -1: break
            # select the max len or all of the training data in the sequence
            # limit the length, seq contains the actual training sequence
        # interacted items
        rated = set(train[u] + valid[u])
        rated.add(0)
        # ground truth item
        neg = []
        for _ in range(sample_nums):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            neg.append(t)
        test_num = len(test[u])
        item_idx = test[u] + neg
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])[0]
        # target distance
        target_ds = predictions[:test_num]
        # sampled results
        sample_d = predictions[test_num:]
        for target_d in target_ds:
            bool_tensor = target_d >= sample_d
            count = torch.sum(bool_tensor).item()
            if count < 10:
                Recall_U += 1
        Recall_U = Recall_U / test_num
        Recall += Recall_U
        Recall_U = 0
        # take the coverage@10 for all users
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    return Recall / valid_user, 0.66
