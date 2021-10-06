"""
-- =============== TODO run combination optimization
pai -name tensorflow180
-Dtables="odps://search_student_dev/tables/F_ui,odps://search_student_dev/tables/F_i,odps://search_student_dev/tables/item_0503_0505_clk_rate_2"

-Dscript="odps://search_student_dev/resources/chaoyu.py"
sda
"""

# import time
from collections import defaultdict
import tensorflow as tf


class Pair:
    def __init__(self, user_ID, item_ID, CTR):
        self.user_ID = int(user_ID)
        self.item_ID = int(item_ID)
        self.CTR = float(CTR)

    def __repr__(self):
        return str([self.user_ID, self.item_ID, self.CTR])


import random as rd
# import time
import numpy as np

tf.app.flags.DEFINE_string("tables", "", "fui, ctr, test_data")
tf.app.flags.DEFINE_integer('is_valid', 1,
                            '1: validation; 0: not validation')
tf.app.flags.DEFINE_string("outputs", "", "tables info saving")

FLAGS = tf.app.flags.FLAGS
# [train_table, test_table]
tables = FLAGS.tables.strip().split(",")

F_ui_name = tables[0]
F_ui_col = "group_id,item_id,_c2"

F_i_name = tables[1]
F_i_col = "item_id,score"
 
test_i_name = tables[2]
test_i_col = "item_id,clk_num"

O = tables[2].split('_')[-3]
 

def tuple2dict(x, mod='item_score'):
    k = len(x[0])
    if mod == 'item_score':
        d = {}
        for i in range(len(x)):
            d[int(x[i][0])] = float(x[i][1])
        return d

    if mod == 'user_item_score':
        pairs = []
        for i in range(len(x)):
            pairs.append(Pair(x[i][0], x[i][1], x[i][2]))
        # pairs = [Pair(pair.split("_")[0], pair.split("_")[1], score)
        #          for pair, score in pairs.items()]
        return pairs
    if mod == 'test':
        pairs = []
        for i in range(len(x)):
            # test data does not have user_id
            pairs.append(Pair(0, x[i][0], x[i][1]))
        return pairs


def load_table(name="odps://search_student_dev/tables/F_ui",
               cols="user_id,item_id,score",
               batch_size=None):
    reader = tf.python_io.TableReader(name, selected_cols=cols)
    if batch_size is None:
        batch_size = reader.get_row_count()  # return 3

    uis = reader.read(batch_size)
    reader.close()

    print('load: {}, cols: {}, num: {}, exp: {}'.format(name, cols,
                                                        batch_size,
                                                        uis[0]))
    return uis


UI_S = load_table(F_ui_name, F_ui_col)
UI_S_2 = tuple2dict(UI_S, 'user_item_score')

I_S = load_table(F_i_name, F_i_col)
I_S_2 = tuple2dict(I_S, 'item_score')

test_I_S = load_table(test_i_name, test_i_col)
test_I_S_2 = tuple2dict(test_I_S, 'test')

 
def TopK(F, K, K_choose, mod='sum'):

    if mod == 'sum':
        score_item = {}
        for user in F:
            for item, socre in F[user].items():
                if item not in score_item:
                    score_item[item] = 0
                score_item[item] += socre

        score_item = [[item, socre] for item, socre in score_item.items()]
        score_item.sort(key=lambda x: x[1], reverse=True)
        return {score_item[i][0] for i in range(K)}, \
               {score_item[i][0] for i in range(K, K_choose)}
    if mod == 'avg':
        score_item_list = defaultdict(list)
        score_item = {}
        for user in F:
            for item, score in F[user].items():
                score_item_list[item].append(score)

        for k, v in score_item:
            if len(v) == 0:
                score_item[k] = 0
            else:
                score_item[k] = sum(v) / len(v)
        score_item.sort(key=lambda x: x[1], reverse=True)
        return {score_item[i][0] for i in range(K)}, \
               {score_item[i][0] for i in range(K, K_choose)}

 

def f(F, S, O):
    obj = 0
    Y = {}

    for user in F:
        put = [[item, score] for item, score in F[user].items()
               if item in S]
        put.sort(key=lambda x: x[1], reverse=True)

        put = put[:min(len(put), O)]

        obj += sum([pair[1] for pair in put])
        Y[user] = [pair[0] for pair in put]

    return obj, Y


def swap_random(S, T):
    S1 = {_ for _ in S}
    T1 = {_ for _ in T}

    i = rd.choice(list(S1))
    j = rd.choice(list(T1))

    S1.remove(i)
    T1.remove(j)
    S1.add(j)
    T1.add(i)

    return S1, T1


def Heuristic_swap_random(M, N, O, K, F,
                          K_choose, times, is_print=True,
                          is_IP=True,
                          epoch=None):
    K_choose = min(K_choose, N)
    # S is select top item, T is candidate item (size_top_item * 2)
    S, T = TopK(F, K, K_choose, mod='sum')
    S0 = {item for item in S}

    obj0, Y0 = f(F, S, O)
    obj_TopK = obj0  # score befor swap
    if is_IP is False:
        return obj0, obj0, S, S

    loop = 0
    if is_print:
        print("\n g-i loop:", loop, "obj_TopK:", obj_TopK)
    flag = True

    while flag:
        flag = False
        for i in range(times):
            loop += 1
            S1, T1 = swap_random(S, T)
            obj1, Y1 = f(F, S1, O)
            if obj0 < obj1:
                S, T = S1, T1
                obj0, Y0 = obj1, Y1
                flag = True
                break
        if is_print:
            print("g-i O: {}, K: {}, loop: {}, obj:{}".format(O, K, loop, obj0))
        if epoch is not None and loop > epoch:
            break

    return obj0, obj_TopK, S, S0


def user_and_item_read(path):
    pairs = pickle.load(open(path, "rb"))
    pairs = [Pair(pair.split("_")[0], pair.split("_")[1], score)
             for pair, score in pairs.items()]
    return pairs


def item_read(path):
    data_train_item = pickle.load(open(path, "rb"))
    data_train_item = {int(k): float(v) for k, v in data_train_item.items()}
    # tmp = load_table('item_score')
    return data_train_item


def data_filter(data_train, data_test, data_train_item):

    num_data_train = len(data_train)
    num_data_test = len(data_test)
    num_data_train_item = len(data_train_item)

    S_item_train = set()
    for pair in data_train:
        S_item_train.add(pair.item_ID)

    S_item_test = set()
    for pair in data_test:
        S_item_test.add(pair.item_ID)

    S_item_train1 = set(data_train_item.keys())
 
    S = S_item_train & S_item_test & S_item_train1
    print(' item set size: {}, {}, {}, {}'.format(len(S), len(S_item_train), len(S_item_test), len(S_item_train1)))

    set_del_data_train = set()
    for i in range(len(data_train)):
        if data_train[i].item_ID not in S:
            set_del_data_train.add(i)

    set_del_data_test = set()
    for i in range(len(data_test)):
        if data_test[i].item_ID not in S:
            set_del_data_test.add(i)

    set_del_data_train_item = set()
    for k in data_train_item:
        if k not in S:
            set_del_data_train_item.add(k)

    data_train = [data_train[i]
                  for i in range(len(data_train)) if i not in set_del_data_train]
    data_test = [data_test[i]
                 for i in range(len(data_test)) if i not in set_del_data_test]
    data_train_item = {k: v for k, v in data_train_item.items()
                       if k not in set_del_data_train_item}
    print("\ndata_train: ", num_data_train, len(data_train))
    print("\ndata_test: ", num_data_test, len(data_test))
    print("\ndata_train_item: ", num_data_train_item, len(data_train_item))
    return data_train, data_test, data_train_item


def data_read_and_filter():
    data_train_item = I_S_2
    data_train = UI_S_2
    data_test = test_I_S_2
    data_train, data_test, data_train_item = data_filter(
        data_train, data_test, data_train_item)
    return data_train, data_test, data_train_item



def user_and_item_count(data_train):
    # t1 = time.perf_counter()
    D_user = {}
    D_item = {}
    for pair in data_train:
        if pair.user_ID not in D_user:
            D_user[pair.user_ID] = set()
        D_user[pair.user_ID].add(pair.item_ID)

        if pair.item_ID not in D_item:
            D_item[pair.item_ID] = set()
        D_item[pair.item_ID].add(pair.user_ID)
    return D_user, D_item, len(D_user), len(D_item)


def F_generate(data_train):
    F = {}
    for pair in data_train:
        if pair.user_ID not in F:
            F[pair.user_ID] = {}
        F[pair.user_ID][pair.item_ID] = pair.CTR
    return F


def S_item_socre_TopK_generate(data_train_item, K):
    S_temp = [[k, v] for k, v in data_train_item.items()]
    S_temp.sort(key=lambda x: x[1], reverse=True)

    return set([S_temp[i][0] for i in range(min(len(S_temp), K))])


def S_test(S, data_test):
    click = sum([pair.CTR for pair in data_test if pair.item_ID in S])
    return click


def hit(s1, data_test, if_debug=False):

    d = {}
    for pair in data_test:
        d[pair.item_ID] = pair.CTR
    d_sorted = sorted(d.items(), key=lambda item: item[1], reverse=True)

    s2 = set()
    for i in range(k):
        s2.add(d_sorted[i][0])

    common = len(s1 & s2)
    if if_debug:
        print('common , {}'.format(common))
        print(s1 & s2)
        print('s1, {}. s2, {} '.format(list(s1)[:5],
                                       list(s2)[:5])
              )
    print('hit@{}: {}/{}, s1 size: {}, s2 size: {}'.format(len(s1), common, len(s1), len(s1), len(s2)))


if __name__ == '__main__':
    print("group-item comb-opt....")
    # 1 load data
    print('data_read_and_filter')
    data_train, data_test, data_train_item = data_read_and_filter()

    # 2 basic info
    print('get basic info')
    D_user, D_item, M, N = user_and_item_count(data_train)

    # 3 F score
    F = F_generate(data_train)
    K = 500
    # O is extract from table_name

    epoch =None# 2000
    if_debug = False
    # t4 = time.perf_counter()
    print('Heuristic_swap_random')
    obj, obj_F_TopK, S, S_F_TopK = Heuristic_swap_random(M,
                                                         N,
                                                         O,
                                                         K,
                                                         F,
                                                         4 * K,
                                                         times=3000,
                                                         is_print=True,
                                                         is_IP=True,
                                                         epoch=epoch
                                                         )
    print('TRAIN: select item on training set based on F_gi')
    print("obj_F_TopK:", obj_F_TopK)
    print("obj:", obj)

    # 4-get top-k item
    print('TEST: eval on test_set')
    print('top-k item eval')
    S_item_socre_TopK = S_item_socre_TopK_generate(data_train_item, K)

    CTR_TopK = S_test(S_F_TopK, data_test)
    CTR = S_test(S, data_test)
    CTR_item_socre_TopK = S_test(S_item_socre_TopK, data_test)

    print("\nCTR_F_TopK: ", CTR_TopK, 'hit: ', hit(S_F_TopK, data_test, if_debug=if_debug))
    print("\nCTR: ", CTR, 'hit: ', hit(S, data_test))
    print("\nCTR_item_socre_TopK: ", CTR_item_socre_TopK, 'hit: ', hit(S_item_socre_TopK, data_test))
    print(tables)
 