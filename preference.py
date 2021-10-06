
# !/usr/bin/env python
# -*- coding: utf-8 -*-
 

import tensorflow as tf
import time
import numpy as np

localtime = time.asctime(time.localtime(time.time()))
print(localtime)

print(tf.__version__)

tf.app.flags.DEFINE_string("tables", "", "tables info including train/test")
tf.app.flags.DEFINE_integer('is_valid', 1,
                            '1: validation; 0: not validation')
tf.app.flags.DEFINE_string("outputs", "", "tables info saving")

FLAGS = tf.app.flags.FLAGS
# [train_table, test_table]
tables = FLAGS.tables.strip().split(",")
print(tables)
# if TRAIN_WRITE_FLAG and VALID_WRITE_FLAG:
outputs = FLAGS.outputs.strip().split(",")
print(outputs)

# ===================================== hyper para
BATCH_SIZE = 1000

EMB_DIM = 32  # for feature embeding
DIM = 32  # for node embedding and clustering embedding
NUM_EPOCH = int(5e4)
LR = 1e-3  # 1.0
L2_REG = 1e-3  # 1.0
KEEP_PROB = 0.2
model_type = 'hete_gmn'  # 'gmn'

early_stop = 1000

ui_flag = 1
gi_flag = 1

ctr_flag = 0
UI_LOG = 0
GI_LOG = 0
CTR_LOG = 0
update_user_emb_via_UI = 0
update_item_emb_via_IU = 0

# new
"""
user fea
    1'id_age:', coalesce(id_age, ''), '#',
    2'id_age_level:', coalesce(id_age_level, ''),
    3'id_gender:', coalesce(id_gender, ''), '#',
    4'prov_name:', coalesce(prov_name, ''), '#',
    5'city_name:', coalesce(city_name, ''), '#',
    6'county_name:', coalesce(county_name, ''), '#',
    7'city_level:', coalesce(city_level, ''), '#',
    8'is_cap:', coalesce(is_cap, ''), '#',
    9'buyer_star_name:', coalesce(buyer_star_name, ''), '#',
    10'tm_level:', coalesce(tm_level, ''), '#',
    11'vip_level_name:', coalesce(vip_level_name, ''), '#',
    12'phone_brand:', coalesce(phone_brand, ''), '#',
    13'phone_model:', coalesce(phone_model, ''), '#',
    14'phone_price_level_prefer:', coalesce(phone_price_level_prefer, ''), '#',
    15'pred_car_brand:', coalesce(pred_car_brand, ''), '#',
    16'pred_career_type:', coalesce(pred_career_type, ''), '#',
    17'pred_has_car:', coalesce(pred_has_car, ''), '#',
    18'pred_has_house:', coalesce(pred_has_house, ''), '#',
    19'pred_life_stage_haschild:', coalesce(pred_life_stage_haschild, ''), '#',
    20'pred_life_stage_married:', coalesce(pred_life_stage_married, ''), '#',
    21'property_hourse_level:', coalesce(property_hourse_level, ''), '#',
    22'vst_days_1m:', coalesce(vst_days_1m, ''), '#',
    23'clt_slr_cnt_2w:', coalesce(clt_slr_cnt_2w, ''), '#',
    24'clt_itm_cnt_2w:', coalesce(clt_itm_cnt_2w, ''), '#',
    25'user_layer_level:', coalesce(user_layer_level, ''), '#',
    26'purchase_total:', coalesce(purchase_total, ''), '#',
    27'os:', coalesce(os, ''), '#',
    28'user_layer:', coalesce(user_layer, ''), '#',
    29'activity:', coalesce(activity, '')
"""

#
user_hash_size_list = [
    121, 9, 2, 51, 834,
    4061, 6, 2, 19, 5,
    9, 7968, 1667, 9, 283,
    10, 2, 2, 2, 2,
    3, 30, 1752, 2222, 9,
    7, 6, 9, 6
    # 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10
]
user_hash_size_list = [int(i * 10) for i in user_hash_size_list]
NUM_USER_ID = int(2e7)
user_hash_size_list = [NUM_USER_ID]  # + user_hash_size_list

"""
item_fea
concat(
    1'mainse_searcher_rank__user_type:', coalesce(mainse_searcher_rank__user_type, ''), '#',
    2'mainse_searcher_rank__category:', coalesce(mainse_searcher_rank__category, ''), '#',
    3'mainse_searcher_rank__cate_level:', coalesce(mainse_searcher_rank__cate_level, ''), '#',
    4'mainse_searcher_rank__cate_level1_id:', coalesce(mainse_searcher_rank__cate_level1_id, ''), '#',
    5'mainse_searcher_rank__shop_level:', coalesce(mainse_searcher_rank__shop_level, ''), '#',
    6'mainse_searcher_rank__sex_c2c:', coalesce(mainse_searcher_rank__sex_c2c, ''), '#',
    7'mainse_searcher_rank__age_level_c2c:', coalesce(mainse_searcher_rank__age_level_c2c, ''), '#',
    8'mainse_searcher_rank__price_level_c2c:', coalesce(mainse_searcher_rank__price_level_c2c, '')
    ) as item_fea
"""


item_hash_size_list = [2, 16722, 4, 1, 1, 4, 7, 8]
item_hash_size_list = [int(i * 10) for i in item_hash_size_list]
NUM_ITEM_ID = int(5e5)
item_hash_size_list = [NUM_ITEM_ID]  # + item_hash_size_list
ALPHA = 0.1  
TAU = 1.0

N_CLUSTER = [0] + [1e3]



print("config: ", ui_flag, gi_flag, ctr_flag,
      update_user_emb_via_UI, update_item_emb_via_IU,
      BATCH_SIZE, EMB_DIM, LR, L2_REG,
      KEEP_PROB, model_type,
      user_hash_size_list, item_hash_size_list,
      ALPHA, TAU, N_CLUSTER)

initializer = tf.contrib.layers.xavier_initializer(uniform=False)
regularizer = tf.contrib.layers.l2_regularizer(L2_REG)


def decode_node_id(info, hash_size, is_hash=False):
    user_hash_size_list
    id_val = tf.decode_csv(info,
                           [[" "]])
    if is_hash:
        id_hash_val = tf.string_to_hash_bucket(id_val, hash_size)
        return id_hash_val
    return id_val


def decode_node_attr(infos, hash_size_list, is_hash=False):
    fea_val_list = [tf.decode_csv(info,
                                  [[" "], [" "]],
                                  ":")[1]
                    for info in infos]
    if is_hash:
        fea_hash_list = [tf.string_to_hash_bucket(i, j)
                         for (i, j) in zip(fea_val_list, hash_size_list)]
        return fea_hash_list
    return fea_val_list


def decode_node_list_attr(infos, node_num, hash_size_list, is_hash=False):
    infos_list = tf.decode_csv(infos,
                               [[" "]] * node_num,
                               chr(3))
    infos_fea_list = [tf.decode_csv(i,
                                    [[' ']] * len(hash_size_list),
                                    '#')
                      for i in infos_list]

    infos_fea_val_list = [decode_node_attr(node, hash_size_list,
                                           is_hash=False)
                          for node in infos_fea_list]
    return_list = [[] for i in range(len(hash_size_list))]

    for x in infos_fea_val_list:
        for idx, val in enumerate(hash_size_list):
            return_list[idx].append(x[idx])

    if is_hash:
        return_hash_list = [
            tf.string_to_hash_bucket(node, hash_size)
            for node, hash_size in zip(return_list, hash_size_list)
        ]
        return return_hash_list


def input_fn_1021(table,
                  selected_cols="user_id,item_id,ui_fea,uu_fea,label",
                  shuffle=True):
    col_num = len(selected_cols.split(','))
    print('input_fn: {}'.format(table))
    print('select col: {}'.format(selected_cols))
    file_queue = tf.train.string_input_producer([table],
                                                num_epochs=NUM_EPOCH,
                                                shuffle=shuffle)

    reader = tf.TableRecordReader(selected_cols=selected_cols)
    keys, values = reader.read_up_to(file_queue,
                                     num_records=BATCH_SIZE)
    default_val = [[' ']] * col_num
    default_val[-1] = [-1.0]
    [user_id, item_id, ui_fea, uu_fea, label] = tf.decode_csv(values, default_val)

    u_id_hash = tf.string_to_hash_bucket(user_id, NUM_USER_ID)
    i_id_hash = tf.string_to_hash_bucket(item_id, NUM_ITEM_ID)

    uu_info_hash = decode_node_list_attr(uu_fea,
                                         5,  # uu neigh
                                         user_hash_size_list,
                                         is_hash=True)
    ui_info_hash = decode_node_list_attr(ui_fea,
                                         5,
                                         item_hash_size_list,
                                         is_hash=True)
    return user_id, item_id, u_id_hash, i_id_hash, label, uu_info_hash, ui_info_hash


def cat_fea_emb_list(fea_list, top=None):
    if top is None:
        return tf.concat(fea_list, axis=-1)
    else:
        return tf.concat(fea_list[:top], axis=-1)


def multi_fea_emb_list(emb_list):
    emb_list_expand = [tf.expand_dims(emb, axis=1) for emb in emb_list]
    return tf.concat(emb_list_expand, axis=1)


def avg_fea_emb_list(fea_list):
    fea_list_expanded = [tf.expand_dims(fea, axis=-1) for fea in fea_list]
    fea_list_concat = tf.concat(fea_list_expanded, axis=-1)
    fea_list_avg = tf.reduce_mean(fea_list_concat, axis=-1)
    return fea_list_avg


def aggregator(node, neigh, type='mean'):
    if type == 'mean':
        return tf.concat([node, tf.reduce_mean(neigh, axis=1)],
                         axis=1)


 
def agg_neigh_id_emb(x_emb, id_emb_mat, xx_info_hash):
    xx_fea_emb_list = [
        tf.nn.embedding_lookup(
            id_emb_mat, xx_info_hash[i])
        for i in range(len(xx_info_hash))
    ]

    xx_emb_list = [tf.transpose(i, [1, 0, 2]) for i in xx_fea_emb_list]
    xx_emb_concat = cat_fea_emb_list(xx_emb_list)

    xx_emb_via_neigh = aggregator(x_emb, xx_emb_concat)
    return xx_emb_via_neigh

 
def model_fn_1021(user_id,
                  item_id,
                  u_id_hash,
                  i_id_hash,

                  batch_y,
                  uu_info_hash,
                  ui_info_hash,
                  keep_prob=None,
                  model_type='gmn'):
    print('model: {} ........'.format(model_type))

    u_id_emb_mat = tf.get_variable('user_id_emb_mat',
                                   [NUM_USER_ID, EMB_DIM],
                                   initializer=initializer)
    i_id_emb_mat = tf.get_variable('item_id_emb_mat',
                                   [NUM_ITEM_ID, EMB_DIM],
                                   initializer=initializer)

    u_id_emb = tf.nn.embedding_lookup(u_id_emb_mat, u_id_hash)
    i_id_emb = tf.nn.embedding_lookup(i_id_emb_mat, i_id_hash)

    batch_y = tf.expand_dims(batch_y, axis=1)

    u_emb_via_uu = agg_neigh_id_emb(u_id_emb, u_id_emb_mat, uu_info_hash)
    u_emb_via_ui = agg_neigh_id_emb(u_id_emb, i_id_emb_mat, ui_info_hash)

    u_emb_for_pred = tf.layers.dense(
        tf.concat([u_emb_via_ui, u_emb_via_uu], axis=1),
        EMB_DIM,
        activation=tf.nn.elu,
        use_bias=True,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name='user_2_cluster'
    )

    i_emb_for_pred = i_id_emb
    # ================= ui pred
    print('ui pred')
    ui_h1 = tf.layers.dense(
        tf.concat([u_emb_for_pred, i_emb_for_pred], axis=1),
        EMB_DIM * 3,
        activation=tf.nn.elu,
        use_bias=True,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name='ui_h1_layer_1'
    )

    ui_h1 = tf.nn.dropout(ui_h1, keep_prob=keep_prob)

    ui_h2 = tf.layers.dense(
        ui_h1,
        EMB_DIM * 3,
        activation=tf.nn.elu,
        use_bias=True,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name='ui_h1_layer_2')

    ui_h2 = tf.nn.dropout(ui_h2, keep_prob=keep_prob)
    ui_pred = tf.layers.dense(ui_h2, 1,
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              name='ui_pred_layer'
                              )
    ui_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_y,
                                                                     logits=ui_pred))
    ui_auc, ui_auc_op = tf.metrics.auc(labels=batch_y,
                                       predictions=tf.nn.sigmoid(ui_pred))
    ui_score = tf.nn.sigmoid(ui_pred)
 



    return user_id, item_id, \
           ui_score, \
           ui_pred, ui_loss, ui_auc, ui_auc_op


# ======================  train
train_infos = input_fn_1021(tables[0], shuffle=True)
print("len of train_infos: {}".format(len(train_infos)))

with tf.variable_scope('model'):
    train_user_id_op, train_item_id_op, \
    train_ui_score_op, \
    train_ui_pred, train_ui_loss, train_ui_auc, train_ui_auc_op, \
    train_gi_pred, train_gi_loss, train_gi_auc, train_gi_auc_op= \
        model_fn_1021(
            *train_infos,
            keep_prob=KEEP_PROB,
            model_type=model_type)

#   ======================== validation =================
if FLAGS.is_valid:
    valid_infos = input_fn_1021(tables[1], shuffle=False)

    with tf.variable_scope('model', reuse=True):
        valid_user_id_op, valid_item_id_op, \
        valid_ui_score_op, \
        valid_ui_pred, valid_ui_loss, valid_ui_auc, valid_ui_auc_op \
            = model_fn_1021(
            *valid_infos, keep_prob=1.0, model_type=model_type)

train_ui_op = tf.train.AdamOptimizer(LR).minimize(train_ui_loss)

 

# saver = tf.train.Saver(tf.global_variables())

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

max_valid_auc = 0.0
# max_ui_auc = 0.0
k = 0
print('start sess....................................')
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
 
    try:
        # train
        for i in range(NUM_EPOCH):
            t1 = time.time()
            if ui_flag:
                train_user_id, train_item_id, train_ui_loss_value, _, _, train_ui_auc_value, train_ui_score = sess.run(
                    [train_user_id_op, train_item_id_op, train_ui_loss, train_ui_op, train_ui_auc_op, train_ui_auc,
                     train_ui_score_op])
                print("{}-TRAIN [ui_LOSS: {:.4f}, ui_AUC: {:.4f}]".format(i,
                                                                          train_ui_loss_value, train_ui_auc_value))
 


    except tf.errors.OutOfRangeError:
        print('done')

    finally:

        coord.request_stop()
        coord.join(threads)
