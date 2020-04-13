#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: model_train.py
    Description:
    
Created by YongBai on 2020/3/18 3:08 PM.
"""
import os
import pandas as pd
import glob
import shutil
import tensorflow as tf
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from keras.utils import multi_gpu_model
from .network import seq_net, load_model
from .model_util import *
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import joblib
import sys
sys.path.append('..')
from data_preprocess import get_train_test_samples, load_seq_feature, get_tn_sample_4train, get_sample_info
from utils import get_config


# reference: https://github.com/GeekLiB/keras/tree/master/keras
def precision(y_true, y_pred):
    '''
    Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    '''
    Calculates the f-measure, the harmonic mean of precision and recall.
    '''
    return fbeta_score(y_true, y_pred, beta=1)


def get_class_weight(y_train):
    cls = np.unique(y_train)
    cls_weight = compute_class_weight('balanced', cls, y_train)
    class_weight_dict = dict(zip(cls, cls_weight))
    return class_weight_dict


def tversky_loss(beta):

    def loss(y_true, y_pred):
        numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
        return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)
    return loss


def train_run(model_out_root_dir, chr='21', epochs=100, batch=32, lr=1e-3,
              filters=16, kernel_size=16, drop=0.2, l2r=1e-4, n_gpu=4, final_model=False):

    # load train sets
    # _pos_df, _ = get_train_test_samples(chr=chr, f_type='tp', reload=True)
    # _neg_df, _ = get_train_test_samples(chr=chr, f_type='tn', reload=True)

    # load
    _neg_df = get_tn_sample_4train(reload=True)
    _pos_df = get_sample_info(reload=True, seq_type='tp', chr=chr)

    pos_df = _pos_df[['Sample_id']].copy()
    pos_df['Label'] = np.ones(len(_pos_df))

    neg_df = _neg_df[['Sample_id']].copy()
    neg_df['Label'] = np.zeros(len(_neg_df))

    train_df = pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)
    train_seq_arr, train_y, _ = _load_seq_features(train_df)

    print('the # of positives and negatives in trains: {} {}'.format(
        len(train_y[train_y == 1]), len(train_y[train_y == 0])))
    train_len, train_win_size, n_train_feat = train_seq_arr.shape

    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        train_seq_arr.astype(np.float32).reshape(-1, train_win_size * n_train_feat)
    ).reshape(-1, train_win_size, n_train_feat)

    if final_model:
        out_fname = 'chr{0}_seqnet_final'.format(chr)
    else:
        out_fname = 'chr{0}_seqnet_trainval'.format(chr)

    # joblib.dump(scaler, os.path.join(model_out_root_dir, 'scaler_train.pkl'))
    joblib.dump(scaler, os.path.join(model_out_root_dir, '{}-scaler.pkl'.format(out_fname)))

    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    base_model = seq_net(train_win_size, n_train_feat, filters=filters, kernel_size=kernel_size,
                         strides=1, pool_size=2, pool_stride=2, drop=drop, kernel_regular_l2=l2r)

    if n_gpu > 1:
        model = multi_gpu_model(base_model, n_gpu)
    else:
        model = base_model

    # output folder
    _model_dir = os.path.join(model_out_root_dir, 'model_weight')
    if not os.path.isdir(_model_dir):
        os.mkdir(_model_dir)
    _tb_dir = os.path.join(model_out_root_dir, 'tb_logs')
    if not os.path.isdir(_tb_dir):
        os.mkdir(_tb_dir)
    _csvlogger_dir = os.path.join(model_out_root_dir, 'model_csvlogger')
    if not os.path.isdir(_csvlogger_dir):
        os.mkdir(_csvlogger_dir)

    # output
    model_fn = os.path.join(_model_dir, '{}.hdf5'.format(out_fname))
    if os.path.exists(model_fn):
        os.remove(model_fn)

    tensorboard_fn = os.path.join(_tb_dir, '{}'.format(out_fname))
    if os.path.isdir(tensorboard_fn):
        shutil.rmtree(tensorboard_fn, ignore_errors=True)

    csvlogger_fn = os.path.join(_csvlogger_dir, '{}'.format(out_fname))
    if os.path.exists(csvlogger_fn):
        os.remove(csvlogger_fn)

    callbacks = [
        # Early stopping definition
        EarlyStopping(monitor='acc' if final_model else 'val_acc', mode='max', patience=10, verbose=1),
        # Decrease learning rate by 0.5 factor
        AdvancedLearnignRateScheduler(monitor='acc' if final_model else 'val_acc',
                                      patience=1, verbose=1, mode='max', decayRatio=0.8),
        # CyclicLR(mode='triangular', base_lr=lr, max_lr=0.1, step_size=6 * (train_len // batch)),
        # Saving best model
        MultiGPUCheckpointCallback(model_fn, base_model=base_model,
                                   monitor='acc' if final_model else 'val_acc', mode='max',
                                   save_best_only=True, verbose=1, save_weights_only=True),
        # histogram_freq=0 because
        # ValueError: If printing histograms, validation_data must be provided, and cannot be a generator
        # set histogram_freq=0 to solve the problem
        TensorBoard(tensorboard_fn, batch_size=batch, histogram_freq=0),
        CSVLogger(csvlogger_fn)
    ]
    # custom_loss = tversky_loss(0.4)  # beta > 0.5 -> precision high
    custom_loss = 'binary_crossentropy'

    model.compile(optimizer=keras.optimizers.Adam(lr=lr),
                  loss=custom_loss,
                  metrics=['accuracy', precision, recall, fmeasure])

    if final_model:
        model.fit(X_train, train_y, epochs=epochs, batch_size=batch, shuffle=True,  # class_weight=get_class_weight(train_y),
                  callbacks=callbacks)
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, train_y, test_size=0.1, random_state=0, stratify=train_y)
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch, shuffle=True,  #class_weight=get_class_weight(train_y),
              callbacks=callbacks, validation_data=(X_val, y_val))


def _load_seq_features(in_df, train=True):

    aneu_conf = get_config()
    seq_out_root_dir = os.path.join(aneu_conf['data_dir']['data_root_dir'], 'corrgc_bams')
    seq_feat_lst = []
    y_label_lst = []
    for index, row in in_df.iterrows():
        sample_id = row['Sample_id']
        feature_fname = glob.glob(os.path.join(seq_out_root_dir, sample_id, '*.features'))[0]
        # logging.info(feature_fname)
        seq_feat_lst.append(load_seq_feature(feature_fname))
        if train:
            y_label = row['Label']
            y_label_lst.append(y_label)

    return np.array(seq_feat_lst), np.array(y_label_lst, dtype=int) if train else None, in_df['Sample_id'].values


def predict_run(model_root_dir, chr='21',
              filters=16, kernel_size=16, drop=0.2, l2r=1e-4, final_model=True):

    """
    this function need to be improved
    :param model_root_dir:
    :param chr:
    :param filters:
    :param kernel_size:
    :param drop:
    :param l2r:
    :param include_top:
    :param final_model:
    :return:
    """
    if final_model:
        out_fname = 'chr{0}_seqnet_final'.format(chr)
    else:
        out_fname = 'chr{0}_seqnet_trainval'.format(chr)

    # load scaler  estimator
    scaler_est_fname = os.path.join(model_root_dir, '{0}-scaler.pkl'.format(out_fname))
    if not os.path.exists(scaler_est_fname):
        sys.exit('Error: scaler not found: {}'.format(scaler_est_fname))
    scaler = joblib.load(scaler_est_fname)

    # load test data
    _, _pos_test = get_train_test_samples(chr=chr, f_type='tp', reload=True)
    _, _neg_test = get_train_test_samples(chr=chr, f_type='tn', reload=True)
    pos_df = _pos_test[['Sample_id']].copy()
    pos_df['Label'] = np.ones(len(_pos_test))

    neg_df = _neg_test[['Sample_id']].copy()
    neg_df['Label'] = np.zeros(len(_neg_test))

    pos_seq_arr, pos_y, _ = _load_seq_features(pos_df)
    neg_seq_arr, neg_y, _ = _load_seq_features(neg_df)

    print('the # of positives in test: {}'.format(len(pos_y)))
    print('the # of negatives in test: {}'.format(len(neg_y)))

    _, in_win_size, n_feat = pos_seq_arr.shape

    x_pos = scaler.transform(
        pos_seq_arr.astype(np.float32).reshape(-1, in_win_size * n_feat)
    ).reshape(-1, in_win_size, n_feat)
    x_neg = scaler.transform(
        neg_seq_arr.astype(np.float32).reshape(-1, in_win_size * n_feat)
    ).reshape(-1, in_win_size, n_feat)

    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_weight_fname = os.path.join(model_root_dir, 'model_weight', '{}.hdf5'.format(out_fname))
    model = load_model(model_weight_fname, inlcude_top=True,
                       train_win_size=in_win_size, n_train_feat=n_feat,
                       filters=filters, kernel_size=kernel_size, drop=drop, l2r=l2r)

    pos_pred = model.predict(x_pos)
    neg_pred = model.predict(x_neg)

    pos_re_arr = np.hstack((pos_pred, np.ones(len(pos_pred)).reshape(-1, 1)))
    neg_re_arr = np.hstack((neg_pred, np.zeros(len(neg_pred)).reshape(-1, 1)))

    re_arr = np.vstack((pos_re_arr, neg_re_arr))

    re_df = pd.DataFrame(data=re_arr, columns=['Y_PRED', 'Y_TRUE'])
    re_out_fname= os.path.join(
        model_root_dir,
        'results/seqnet-test-pred-chr{0}-{1}.csv'.format(chr, 'final' if final_model else 'trainval'))
    if os.path.exists(re_out_fname):
        os.remove(re_out_fname)
    re_df.to_csv(re_out_fname, sep='\t', index=False)


def cal_seq_feat_net(model_root_dir, chr='21', seq_type='tp',
              filters=16, kernel_size=16, drop=0.2, l2r=1e-4, final_model=True):

    # model file name
    if final_model:
        out_fname = 'chr{0}_seqnet_final'.format(chr)
    else:
        out_fname = 'chr{0}_seqnet_trainval'.format(chr)

    # load scaler  estimator
    scaler_est_fname = os.path.join(model_root_dir, '{0}-scaler.pkl'.format(out_fname))
    if not os.path.exists(scaler_est_fname):
        sys.exit('Error: scaler not found: {}'.format(scaler_est_fname))
    scaler = joblib.load(scaler_est_fname)

    # load data
    if seq_type == 'tn':
        _data_df = get_tn_sample_4train(reload=True)
    else:
        _data_df = get_sample_info(reload=True, seq_type=seq_type, chr=chr)

    data_df = _data_df[['Sample_id']].copy()
    data_seq_arr, _, sample_ids = _load_seq_features(data_df, train=False)

    train_len, in_win_size, n_feat = data_seq_arr.shape
    print('the # of samples in final seq feature: {}'.format(train_len))

    data_scaled = scaler.transform(
        data_seq_arr.astype(np.float32).reshape(-1, in_win_size * n_feat)
    ).reshape(-1, in_win_size, n_feat)

    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_weight_fname = os.path.join(model_root_dir, 'model_weight', '{}.hdf5'.format(out_fname))
    model = load_model(model_weight_fname, inlcude_top=False,
                       train_win_size=in_win_size, n_train_feat=n_feat,
                       filters=filters, kernel_size=kernel_size, drop=drop, l2r=l2r)

    features = model.predict(data_scaled)
    col_names = ['SEQ_F_' + str(i) for i in range(features.shape[1])]
    out_df = pd.DataFrame(data=features, columns=col_names)
    out_df['Sample_id'] = sample_ids

    feat_out_fname = os.path.join(model_root_dir, 'results/{0}_{1}.seqfeature'.format(out_fname, seq_type))
    if os.path.exists(feat_out_fname):
        os.remove(feat_out_fname)
    out_df.to_csv(feat_out_fname, sep='\t', index=False)



