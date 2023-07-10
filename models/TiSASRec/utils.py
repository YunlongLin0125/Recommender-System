import sys
import copy
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue
from main import MODEL_NAMES, LOSS_FUNCTIONS, DENSE_ALL_ACTION, DENSE_ALL_PLUS_PLUS, DENSE_ALL_PLUS, INTEGRATED
from main import SASREC_SAMPLED, NORMAL_SASREC, ALL_ACTION
from main import BCE, SAMPLED_SOFTMAX


def random_neq(l, r, s, num=1):
    ts = []
    for i in range(num):
        t = np.random.randint(l, r)
        while t in s or t in ts:
            t = np.random.randint(l, r)
        ts.append(t)
    # if len(ts) == 1:
    #     return ts[0]
    if num == 1:
        return ts[0]
    else:
        return ts


def computeRePos(time_seq, time_span):
    # the reserved timestamp sequence
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            # clipped matrix
            time_matrix[i][j] = min(time_span, span)
    # return time interval matrix
    return time_matrix


def Relation(user_train, usernum, maxlen, time_span):
    # compute relation matrix
    data_train = dict()
    for user in tqdm(range(1, usernum + 1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1:
                break
        data_train[user] = computeRePos(time_seq, time_span)
        # calculate the relation matrix (time interval matrix
        # between the items in the sequence) for each user
    return data_train


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, relation_matrix, result_queue, SEED):
    def sample(user):

        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1][0]

        idx = maxlen - 1
        ts = set(map(lambda x: x[0], user_train[user]))
        for i in reversed(user_train[user][:-1]):
            # i : (item, timestamp)
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break
        time_matrix = relation_matrix[user]
        return user, seq, time_seq, time_matrix, pos, neg

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))


def sample_function_input_target(user_train, train_target, usernum, itemnum,
                                 batch_size, maxlen, relation_matrix,
                                 result_queue, args, SEED):
    def sample(user):
        idx = maxlen - 1
        target_seq = train_target[user]
        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        sample_actions = 32
        target_seq = [x[0] for x in target_seq]
        # select all item ids
        if len(target_seq) > sample_actions:
            target_seq = random.sample(target_seq, k=sample_actions)
        elif len(target_seq) < sample_actions:
            target_seq = random.choices(target_seq, k=sample_actions)
        ts = set(map(lambda x: x[0], user_train[user]))
        ts.update(target_seq)
        if args.model == ALL_ACTION:
            num_pos = len(target_seq)
        elif args.model == DENSE_ALL_ACTION:
            num_pos = 1
        elif args.model == DENSE_ALL_PLUS:
            num_pos = len(target_seq)
        elif args.model == DENSE_ALL_PLUS_PLUS:
            num_pos = len(target_seq) + 1
        elif args.model == INTEGRATED:
            num_pos = 2
        else:
            num_pos = 1
            print("wrong model")
            quit()
        num_neg = 10
        pos = np.zeros([maxlen, num_pos], dtype=np.int32)
        neg = np.zeros([maxlen, num_neg], dtype=np.int32)
        if args.model in [ALL_ACTION, DENSE_ALL_ACTION, DENSE_ALL_PLUS]:
            for i in reversed(user_train[user]):
                # i : (item, timestamp)
                seq[idx] = i[0]
                time_seq[idx] = i[1]
                if args.model == ALL_ACTION:
                    if idx == maxlen - 1:
                        pos[idx] = target_seq
                        neg[idx] = random_neq(1, itemnum + 1, ts, num_neg)
                elif args.model == DENSE_ALL_ACTION:
                    pos[idx] = random.sample(target_seq, 1)[0]
                    neg[idx] = random_neq(1, itemnum + 1, ts, num_neg)
                elif args.model == DENSE_ALL_PLUS:
                    pos[idx] = target_seq
                    neg[idx] = random_neq(1, itemnum + 1, ts, num_neg)
                idx -= 1
                if idx == -1:
                    break
        elif args.model in [DENSE_ALL_PLUS_PLUS, INTEGRATED]:
            # model: connections include between the two parts and also the next item prediction.
            nxt = user_train[user][-1][0]
            for i in reversed(user_train[user][:-1]):
                # fill the seq with all interacted except the last item as the target action
                seq[idx] = i[0]
                time_seq[idx] = i[1]
                if args.model == DENSE_ALL_PLUS_PLUS:
                    if nxt != 0:
                        pos[idx] = [nxt] + target_seq
                        neg[idx] = random_neq(1, itemnum + 1, ts, num_neg)
                elif args.model == INTEGRATED:
                    if nxt != 0:
                        random_target = random.sample(target_seq, 1)
                        pos[idx] = [nxt] + random_target
                        neg[idx] = random_neq(1, itemnum + 1, ts, num_neg)
                nxt = i[0]
                idx -= 1
                if idx == -1:
                    break
        time_matrix = relation_matrix[user]
        return user, seq, time_seq, time_matrix, pos, neg

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, relation_matrix, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      relation_matrix,
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


class WarpSamplerInputTarget(object):
    def __init__(self, user_input, user_target, usernum, itemnum, relation_matrix, args, batch_size=64, maxlen=10,
                 n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_input_target, args=(user_input,
                                                                   user_target,
                                                                   usernum,
                                                                   itemnum,
                                                                   batch_size,
                                                                   maxlen,
                                                                   relation_matrix,
                                                                   self.result_queue,
                                                                   args,
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


def sample_function_input_target_byT(user_train, train_target, train_ids, usernum, itemnum,
                                     batch_size, maxlen, relation_matrix, result_queue, args, SEED):
    def sample(user):
        # randomly select a trained used user index
        idx = maxlen - 1
        num_neg = 10
        sample_actions = 16
        target_seq = [x[0] for x in train_target[user]]
        if args.model in [DENSE_ALL_ACTION]:
            target_seq = train_target[user]
        else:
            if len(target_seq) > sample_actions:
                target_seq = random.sample(target_seq, k=sample_actions)
            if len(target_seq) < sample_actions:
                target_seq = random.choices(target_seq, k=sample_actions)
        ts = set(map(lambda x: x[0], user_train[user]))
        ts.update(target_seq)
        if args.model == ALL_ACTION:
            num_pos = len(target_seq)
        elif args.model == NORMAL_SASREC:
            num_pos = 1
        elif args.model == SASREC_SAMPLED:
            num_pos = len(target_seq)
        elif args.model == DENSE_ALL_ACTION:
            num_pos = 1
        elif args.model == DENSE_ALL_PLUS:
            num_pos = len(target_seq)
        elif args.model == DENSE_ALL_PLUS_PLUS:
            num_pos = len(target_seq) + 1
        elif args.model == INTEGRATED:
            num_pos = 2
        else:
            num_pos = 1
            print("wrong model")
            quit()
        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen, num_pos], dtype=np.int32)
        neg = np.zeros([maxlen, num_neg], dtype=np.int32)
        if args.model in [ALL_ACTION, DENSE_ALL_ACTION, DENSE_ALL_PLUS]:
            # model: connections only between the input sequence and the target sequence
            for i in reversed(user_train[user]):
                # fill the seq with all interacted except the last item as the target action
                seq[idx] = i[0]  # [x,x,x,x,x,x,#]
                time_seq[idx] = i[1]
                # feed pos
                if args.model == ALL_ACTION:
                    if idx == maxlen - 1:
                        pos[idx] = target_seq
                        neg[idx] = random_neq(1, itemnum + 1, ts, num_neg)
                elif args.model == DENSE_ALL_ACTION:
                    pos[idx] = random.sample(target_seq, 1)[0]
                    neg[idx] = random_neq(1, itemnum + 1, ts, num_neg)
                elif args.model == DENSE_ALL_PLUS:
                    pos[idx] = target_seq
                    neg[idx] = random_neq(1, itemnum + 1, ts, num_neg)
                idx -= 1
                if idx == -1:
                    break
        elif args.model in [DENSE_ALL_PLUS_PLUS, INTEGRATED]:
            # model: connections include between the two parts and also the next item prediction.
            nxt = user_train[user][-1][0]
            for i in reversed(user_train[user][:-1]):
                # fill the seq with all interacted except the last item as the target action
                seq[idx] = i[0]  # [x,x,x,x,x,x,#]
                time_seq[idx] = i[1]
                if args.model == DENSE_ALL_PLUS_PLUS:
                    if nxt != 0:
                        pos[idx] = [nxt] + target_seq
                        neg[idx] = random_neq(1, itemnum + 1, ts, num_neg)
                elif args.model == INTEGRATED:
                    if nxt != 0:
                        random_target = random.sample(target_seq, 1)
                        pos[idx] = [nxt] + random_target
                        neg[idx] = random_neq(1, itemnum + 1, ts, num_neg)
                nxt = i[0]
                idx -= 1
                if idx == -1:
                    break
        elif args.model in [NORMAL_SASREC, SASREC_SAMPLED]:
            num_neg = 1
            seq = np.zeros([maxlen], dtype=np.int32)
            time_seq = np.zeros([maxlen], dtype=np.int32)
            pos = np.zeros([maxlen], dtype=np.int32)
            neg = np.zeros([maxlen, num_neg], dtype=np.int32)
            nxt = (user_train[user] + train_target[user])[-1][0]
            ts = [x[0] for x in (user_train[user] + train_target[user])]
            for i in reversed((user_train[user] + train_target[user])[:-1]):
                # fill the seq with all interacted except the last item as the target action
                # i started by user_train[user][-2]
                seq[idx] = i[0]
                time_seq[idx] = i[1]
                pos[idx] = nxt
                if nxt != 0:
                    neg[idx] = random_neq(1, itemnum + 1, ts, num_neg)
                nxt = i[0]
                idx -= 1
                if idx == -1:
                    break
        time_matrix = relation_matrix[user]
        return user, seq, time_seq, time_matrix, pos, neg

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = random.choice(train_ids)
            while len(user_train[user]) <= 1:
                user = random.choice(train_ids)
            one_batch.append(sample(user))
        result_queue.put(zip(*one_batch))


class WarpSamplerInputTarget_byT(object):
    def __init__(self, user_input, user_target, train_ids, usernum, itemnum, relation_matrix, args, batch_size=64
                 , maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_input_target_byT, args=(user_input,
                                                                       user_target,
                                                                       train_ids,
                                                                       usernum,
                                                                       itemnum,
                                                                       batch_size,
                                                                       maxlen,
                                                                       relation_matrix,
                                                                       self.result_queue,
                                                                       args,
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


def timeSlice(time_set):
    """
    This function takes in a set of times (presumably timestamps), and it creates a dictionary
    that maps each timestamp to an integer representation of it,
    relative to the smallest timestamp in the set.
    """
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:  # float as map key?
        time_map[time] = int(round(float(time - time_min)))
    return time_map


def cleanAndsort(User, time_map):
    """
    This function takes in the User dictionary and the time_map dictionary (created by the timeSlice function).
    The User dictionary is assumed to be a nested dictionary where the first key is a user ID,
    and the value is a list of tuples containing item IDs and their associated timestamps.
    """
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        # u, (i, timestamp)
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u + 1
    for i, item in enumerate(item_set):
        item_map[item] = i + 1
    # create the map for user and item
    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])
        # sorted by timestamp
    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]]], items))
    # map the item and timestamp and userid for each user.
    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        # get all timestamps
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                # calculate each time interval to find the minimum time interval of each user
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
            # find the minimum time interval of each user
        time_min = min(time_list)
        # minimum time of the current user. Why need?
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1)], items))
        # get scaled absolute time (- time min)
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))
        # find the max timestamp for each user
    return User_res, len(user_set), len(item_set), max(time_max)


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    print('Preparing data...')
    f = open('data/%s.txt' % fname, 'r')
    time_set = set()
    # data filtering
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        if user_count[u] < 5 or item_count[i] < 5:  # hard-coded
            continue
        time_set.add(timestamp)
        # find all timestamps
        User[u].append([i, timestamp])
        # include the time feature
    f.close()
    time_map = timeSlice(time_set)
    # create time map: {real timestamp: mapped_time} (timestamp - time_min)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)
    # map the time to scaled time
    # mapped User dict
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
    print('Preparing done...')
    return [user_train, user_valid, user_test, usernum, itemnum, timenum]


def data_partition_window_InputTarget_byP(fname, valid_percent, test_percent, train_percent):
    User = defaultdict(list)
    print('Preparing data...')
    f = open('data/%s.txt' % fname, 'r')
    time_set = set()
    # data filtering
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        if user_count[u] < 5 or item_count[i] < 5:  # hard-coded
            continue
        time_set.add(timestamp)
        # find all timestamps
        User[u].append([i, timestamp])
        # include the time feature
    f.close()
    time_map = timeSlice(time_set)
    # create time map: {real timestamp: mapped_time} (timestamp - time_min)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)
    print('Clean Data Done')
    if valid_percent + test_percent > 0.6:
        print('the percent you select for val/test are too high')
        return None
    valid_start = 1 - valid_percent - test_percent
    test_start = 1 - test_percent
    train_start = 1 - train_percent
    user_input = {}
    user_target = {}
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            # continue
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
                # deal with some specific situation
                user_train[user] = User[user]
                split_index = int(len(user_train) * train_start)
                user_input[user] = User[user][:split_index]
                user_target[user] = User[user][split_index:]
                if not user_target[user]:
                    user_target[user] = [User[user][-1]]
                    user_input[user] = User[user][:-1]
                user_valid[user] = []
                user_test[user] = []
            else:
                train_seq = User[user][: valid_index]
                valid_seq = User[user][valid_index: test_index]
                test_seq = User[user][test_index:]
                train_seq_length = len(train_seq)
                split_index = int(train_seq_length * train_start)
                # split the input and the target
                input_seq = train_seq[:split_index]
                user_input[user] = []
                user_input[user] += input_seq
                # store the input seq
                target_seq = train_seq[split_index:]
                # get the current target window
                user_target[user] = []
                user_target[user] += target_seq
                # store the target sequence
                # split the whole sequence to train/valid/test
                user_train[user] = []
                user_train[user] += train_seq
                user_valid[user] = []
                user_valid[user] += valid_seq
                user_test[user] = []
                user_test[user] += test_seq
    return [user_input, user_target, user_train, user_valid, user_test, usernum, itemnum, timenum]


def ScaleTime(User_filted, time_map):
    User_res = dict()
    for user, items in User_filted.items():
        User_res[user] = list(map(lambda x: [x[0], time_map[x[1]]], items))
    # only map the time_map
    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        # get all timestamps
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                # calculate each time interval to find the minimum time interval of each user
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
            # find the minimum time interval of each user
        time_min = min(time_list)
        # minimum time of the current user. Why need?
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1)], items))
        # get scaled absolute time (- time min)
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))
        # find the max timestamp for each user
    return User_res, max(time_max)


def data_partition_window_InputTarget_byT(f_train, f_target):
    usernum = 0
    itemnum = 0
    time_set = set()
    user_input = defaultdict(list)
    user_target = defaultdict(list)
    train_split = 0.7
    valid_split = 0.85
    f = open('data/%s.txt' % f_train, 'r')
    print('Preparing data...')
    # read the input sequence
    for line in f:
        try:
            u, i, timestamp = line.rstrip().split(' ')
        except:
            u, i, rating, timestamp = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        time_set.add(timestamp)
        # find all timestamps
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        user_input[u].append([i, timestamp])
        # include the time feature
    f.close()
    print("Input Read Done")
    time_map = timeSlice(time_set)
    # create time map: {real timestamp: mapped_time} (timestamp - time_min)
    user_input, timenum = ScaleTime(user_input, time_map)
    print("Time Scaled Done!")
    f = open('data/%s.txt' % f_target, 'r')
    # read from the target window
    for line in f:
        try:
            u, i, timestamp = line.rstrip().split(' ')
        except:
            u, i, rating, timestamp = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        # find all timestamps
        itemnum = max(i, itemnum)
        user_target[u].append([i, timestamp])

    # generate user_ids for train/val/test
    print("Target Read Done")
    users = list(user_input.keys())
    rng = random.Random(1)
    rng.shuffle(users)
    # Split the keys into train, valid, test
    train_users = users[:int(usernum * train_split)]
    valid_users = users[int(usernum * train_split):int(usernum * valid_split)]
    test_users = users[int(usernum * valid_split):]
    print("Data processing Done")
    return [user_input, user_target, usernum, itemnum, timenum,
            train_users, valid_users, test_users]


def evaluate_window(model, dataset, args, eval_type='valid'):
    if args.model not in [NORMAL_SASREC, SASREC_SAMPLED]:
        [_, _, train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)
    else:
        [_, _, train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)
    # train.shape = [user_num, seq_num, (itemid, timestamp)]
    Recall = 0.0
    Recall_U = 0.0
    valid_user = 0.0
    sample_nums = 500
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if eval_type == 'valid':
            input_seq = train[u]
            target_seq = [x[0] for x in valid[u]]
            # target sequence only includes the item indexes
        else:
            input_seq = train[u] + valid[u]
            target_seq = [x[0] for x in test[u]]
        if len(input_seq) < 1 or len(target_seq) < 1:
            continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(input_seq):
            # fill the input sequence from the list tail
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1:
                break
        rated = set(map(lambda x: x[0], input_seq))
        rated.update(target_seq)
        rated.add(0)
        neg = []
        for _ in range(sample_nums):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            neg.append(t)
        target_num = len(target_seq)
        item_idx = target_seq + neg
        time_matrix = computeRePos(time_seq, args.time_span)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], item_idx]])[0]
        # target distance
        target_ds = predictions[:target_num]
        # sampled results
        sample_d = predictions[target_num:]
        for target_d in target_ds:
            bool_tensor = target_d >= sample_d
            count = torch.sum(bool_tensor).item()
            if count < 10:
                Recall_U += 1
        Recall_U = Recall_U / target_num
        Recall += Recall_U
        Recall_U = 0
        # take the coverage@10 for all users
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    return Recall / valid_user, 0.66


def evaluate_window_byT(model, dataset, args, eval_type='valid'):
    [user_input, user_target, usernum, itemnum, timenum, _, valid_users, test_users] = dataset
    # train.shape = [user_num, seq_num, (itemid, timestamp)]
    Recall = 0.0
    Recall_U = 0.0
    valid_user = 0.0
    sample_nums = 500
    if eval_type == 'valid':
        users = valid_users
    elif eval_type == 'test':
        users = test_users
    else:
        users = None
        print('Unknown Evaluation')
        quit()
    for u in users:
        input_seq = user_input[u]
        target_seq = [x[0] for x in user_target[u]]
        if len(input_seq) < 1 or len(target_seq) < 1:
            continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(input_seq):
            # fill the input sequence from the list tail
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1:
                break
        rated = set(map(lambda x: x[0], input_seq))
        rated.update(target_seq)
        rated.add(0)
        neg = []
        for _ in range(sample_nums):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            neg.append(t)
        target_num = len(target_seq)
        item_idx = target_seq + neg
        time_matrix = computeRePos(time_seq, args.time_span)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], item_idx]])[0]
        # target distance
        target_ds = predictions[:target_num]
        # sampled results
        sample_d = predictions[target_num:]
        for target_d in target_ds:
            bool_tensor = target_d >= sample_d
            count = torch.sum(bool_tensor).item()
            if count < 10:
                Recall_U += 1
        Recall_U = Recall_U / target_num
        Recall += Recall_U
        Recall_U = 0
        # take the coverage@10 for all users
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    return Recall / valid_user, 0.66
