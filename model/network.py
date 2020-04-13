#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: network.py
    Description:

Created by YongBai on 2020/3/18 2:09 PM.
"""
import os
import sys
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Flatten, Dropout, MaxPooling1D, Activation, BatchNormalization
from keras.layers import Add, Multiply, GlobalAveragePooling1D, Reshape, LSTM, Lambda, Bidirectional, AveragePooling1D
from keras.regularizers import l2
from .attention_layer import Attention


def _bn_relu(x, name=None):
    """
    Helper to build a BN -> relu block
    :param x:
    :param name:
    :return:
    """
    norm = BatchNormalization(name=name + '_bn')(x)
    return Activation("relu", name=name + '_relu')(norm)


def basic_residual_unit(**kwargs):
    """
    Helper to build a conv1d->BN->relu residual unit
    :param res_params:
    :return:
    """

    filters = kwargs['filters']
    b_name = kwargs['name']
    kernel_size = kwargs.setdefault('kernel_size', 16)
    strides = kwargs.setdefault('strides', 1)
    l2r = kwargs.setdefault('l2r', 1e-4)

    def f(x):
        x = Conv1D(filters=filters,
                   kernel_size=kernel_size,
                   padding='same',
                   strides=strides, kernel_regularizer=l2(l2r) if l2r is not None else None,
                   kernel_initializer='he_normal', name=b_name + '_conv')(x)

        return _bn_relu(x, name=b_name)

    return f


def se_block(se_input, ratio=16, name=None):
    """
    Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. CVPR(2018)
    Implementation adapted from
        https://github.com/titu1994/keras-squeeze-excite-network/blob/master/se.py
    :param se_input:
    :param ratio:
    :param name
    :return:
    """

    init = se_input
    filters = init._keras_shape[-1]
    se_shape = (1, filters)

    se = GlobalAveragePooling1D(name=name + '_gap')(init)
    se = Reshape(se_shape, name=name + '_reshape')(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False, name=name + '_dense_relu')(se)
    se = Dense(filters, activation='sigmoid', use_bias=False, name=name + '_dense_sigmoid')(se)

    x = Multiply(name=name + '_multiply')([init, se])
    return x


def spatial_se_block(ses_input, name=None):
    """
    Create a spatial squeeze-excite block
    Ref:
        [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks]
    (https://arxiv.org/abs/1803.02579)
    :param ses_input:
    :param name
    :return:
    """

    se = Conv1D(1, 1, activation='sigmoid', use_bias=False, name=name + '_conv')(ses_input)
    x = Multiply(name=name + '_multiply')([ses_input, se])
    return x


def cse_sse_block(se_input, ratio=16, name=None):
    """
    Create a spatial squeeze-excite block
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks]
        (https://arxiv.org/abs/1803.02579)
    :param se_input:
    :param ratio:
    :param name
    :return:
    """

    cse = se_block(se_input, ratio, name=name + '_se')
    sse = spatial_se_block(se_input, name=name + '_sse')

    x = Add(name=name + '_sse_add')([cse, sse])
    return x


def simple_residual_block(input_x, **kwargs):
    filters = kwargs['filters']
    kernel_size = kwargs.setdefault('kernel_size', 16)
    strides = kwargs.setdefault('strides', 1)
    drop = kwargs.setdefault('drop', 0.2)
    l2r = kwargs.setdefault('l2r', 1e-4)

    pool_size = kwargs['pool_size']
    pool_stride = kwargs['pool_stride']
    name = kwargs['name']

    # left branch
    x1 = Conv1D(filters=filters,
                kernel_size=kernel_size,
                padding='same',
                strides=strides,
                kernel_regularizer=l2(l2r) if l2r is not None else None,
                kernel_initializer='he_normal')(input_x)
    x1 = Dropout(drop)(x1)
    x1 = _bn_relu(x1, name=name + '_br_0')
    x1 = Conv1D(filters=filters,
                kernel_size=kernel_size,
                padding='same',
                strides=strides,
                kernel_regularizer=l2(l2r) if l2r is not None else None,
                kernel_initializer='he_normal')(x1)
    x1 = Dropout(drop)(x1)
    x1 = _bn_relu(x1, name=name + '_br_1')
    # x1 = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(x1)
    x1 = AveragePooling1D(pool_size=pool_size, strides=pool_stride)(x1)

    # right branch
    # x2 = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(input_x)
    x2 = AveragePooling1D(pool_size=pool_size, strides=pool_stride)(input_x)

    x1 = cse_sse_block(x1, name=name + '_cse')
    # x1 = se_block(x1, ratio=16, name=name + '_se')
    x = keras.layers.add([x1, x2])
    del x1, x2
    return x


def seq_net(win_size, n_in_feat, filters=16, kernel_size=16, strides=1, pool_size=2,
            pool_stride=2, drop=0.2, kernel_regular_l2=1e-4):  #

    input_x = Input(shape=(win_size, n_in_feat), name='input')
    x = basic_residual_unit(filters=filters, kernel_size=kernel_size, strides=strides, l2r=kernel_regular_l2,
                            name='start_basic_residual')(input_x)

    for i in range(3):
        x = simple_residual_block(x, filters=filters, kernel_size=kernel_size, strides=strides,
                                  pool_size=pool_size, pool_stride=pool_stride, drop=drop,
                                  name='mid_residual_' + str(i))
    # x = _bn_relu(x, name='end_br')
    # x = BatchNormalization(name='end_norm')(x)
    # x = Bidirectional(LSTM(8, activation='relu', return_sequences=True, dropout=drop, recurrent_dropout=drop),
    #                   merge_mode='ave')(x)
    # x = LSTM(8, return_sequences=True, dropout=drop, recurrent_dropout=drop)(x)
    x = basic_residual_unit(filters=8, kernel_size=kernel_size, strides=strides, l2r=kernel_regular_l2,
                            name='end_basic_residual')(x)
    # x = _bn_relu(x, name='end_br')
    x = GlobalAveragePooling1D(name='end_last_global_avg_pool')(x)
    # x = Lambda(lambda l: l / 4, name='logit_temperature')(x)  # this may decrease acc
    # x = Dense(fc_size, activation='relu', name='end_dense_relu')(x)
    # logits = Dense(n_out_class, name='logits')(x)
    # logits_t = Lambda(lambda l: l / temperature, name='logit_temperature')(logits)
    # out = Activation('sigmoid', name='model_out_prob')(logits)
    # x = Attention(x._keras_shape[-2], name='final_attention')(x)
    x = Dense(1, activation='sigmoid', name='model_out_prob')(x)

    model = Model(inputs=input_x, outputs=x, name='seq_net')

    return model


def load_model(model_weight, inlcude_top=True,
               train_win_size=85, n_train_feat=9, filters=16, kernel_size=16, drop=0.2, l2r=1e-4):

    # model structure
    model = seq_net(train_win_size, n_train_feat, filters=filters, kernel_size=kernel_size,
                         strides=1, pool_size=2, pool_stride=2, drop=drop, kernel_regular_l2=l2r)
    # load weight
    if not os.path.exists(model_weight):
        sys.exit('Error: model weight not found: {}'.format(model_weight))
    model.load_weights(model_weight)

    if inlcude_top:
        return model
    else:
        layer_name = 'end_last_global_avg_pool'
        return Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

