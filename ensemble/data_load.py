#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: data_load.py
    Description:
    
Created by YongBai on 2020/3/20 5:35 PM.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import sys
sys.path.append('..')
from data_preprocess import get_train_test_samples, get_sample_info
from utils import get_config


def extract_phenotype_feature(in_df):

    out_cols = ['SAMPLE_NUM', 'ID_NUMBER', 'HEIGHT', 'WEIGHT', 'GESTATIONAL_WEEKS', 'BLOOD_DATE', 'FF']

    out_df = in_df[out_cols].copy()
    # out_df.loc[:, 'AGE'] = 0.0
    # out_df.loc[:, 'GESTATIONAL_WEEKS_D'] = 0.0

    # calculate AGE
    def getdob_fromid(x):
        y = x[6: 6+8]
        return datetime.strptime(y, '%Y%m%d')

    dob = out_df['ID_NUMBER'].apply(lambda x: getdob_fromid(x))
    abs_diff_time = pd.to_datetime(out_df['BLOOD_DATE']) - dob
    out_df['AGE'] = abs_diff_time.dt.total_seconds() / (60.0 * 60.0 * 24 * 365.25)
    # age = abs_diff_time.dt.total_seconds()/(60.0 * 60.0 * 24 * 365.25)
    # out_df.loc[:, 'AGE'] = age

    # calculate GESTATIONAL_WEEKS
    def get_gw(x):
        pat = re.compile(r'^([0-9]{0,3})w?\+?([0-9]{0,2}).*$', flags=re.IGNORECASE)
        # print(x.strip())
        y = re.findall(pat, x.strip())[0]
        try:
            return float(y[0]) + float(y[1])/7 if len(y) == 2 and y[1].strip() != '' else float(y[0])
        except Exception as e:
            sys.exit('Error: {}'.format(str(e)))

    out_df['GESTATIONAL_WEEKS_D'] = out_df['GESTATIONAL_WEEKS'].apply(lambda x: get_gw(x))
    # g_week = out_df['GESTATIONAL_WEEKS'].apply(lambda x: get_gw(x))
    # out_df.loc[:, 'GESTATIONAL_WEEKS_D'] = g_week
    # out_df = out_df[(out_df['HEIGHT'] > 25.0) & (out_df['WEIGHT'] > 25.0)]

    out_df.loc[out_df['HEIGHT'] < 25.0, 'HEIGHT'] = np.nan
    out_df.loc[out_df['WEIGHT'] < 25.0, 'WEIGHT'] = np.nan

    re_cols = ['SAMPLE_NUM', 'AGE', 'HEIGHT', 'WEIGHT', 'GESTATIONAL_WEEKS_D', 'FF']
    return out_df[re_cols].copy()


def extract_seq_feature(seq_type='tp', chr='21'):
    data_root_dir = get_config()['data_dir']['data_root_dir']
    seq_fname = os.path.join(data_root_dir,
                             'model_update/results/chr{}_seqnet_final_{}.seqfeature'.format(chr, seq_type))
    return pd.read_csv(seq_fname, sep='\t')


def separate_df(in_df, sep_colsnames):

    t_df = in_df.copy()
    sub_df = t_df[sep_colsnames]
    main_df = t_df.drop(columns=sep_colsnames)
    main_colnames = list(main_df.columns)
    main_values = main_df.values

    return sub_df, main_colnames, main_values


def read_feature_all(reload=False, chr='21'):

    data_root_dir = get_config()['data_dir']['data_root_dir']
    train_feature_out_fname = os.path.join(data_root_dir,
                                           'model_update/results/chr{0}_normalized_ens_train.allfeature'.format(chr))
    test_feature_out_fname = os.path.join(data_root_dir,
                                          'model_update/results/chr{0}_normalized_ens_test.allfeature'.format(chr))
    if reload:
        if not os.path.exists(train_feature_out_fname):
            sys.exit('Error: {} not found.'.format(train_feature_out_fname))
        if not os.path.exists(test_feature_out_fname):
            sys.exit('Error: {} not found.'.format(test_feature_out_fname))
        train_df = pd.read_csv(train_feature_out_fname, sep='\t')
        test_df = pd.read_csv(test_feature_out_fname, sep='\t')
        return train_df, test_df

    _tp_train_df, _tp_test_df = get_train_test_samples(chr=chr, f_type='tp', reload=True)
    _tn_train_df, _tn_test_df = get_train_test_samples(chr=chr, f_type='tn', reload=True)

    print('the orignal # of samples in training set: {}'.format(len(_tp_train_df) + len(_tn_train_df)))
    print('the orignal # of samples in testing set: {}'.format(len(_tp_test_df) + len(_tn_test_df)))

    tp_pheno_feat_train_df = extract_phenotype_feature(_tp_train_df)
    tp_pheno_feat_train_df.loc[:, 'LABEL'] = 1
    tp_pheno_feat_test_df = extract_phenotype_feature(_tp_test_df)
    tp_pheno_feat_test_df.loc[:, 'LABEL'] = 1

    tn_pheno_feat_train_df = extract_phenotype_feature(_tn_train_df)
    tn_pheno_feat_train_df.loc[:, 'LABEL'] = 0
    tn_pheno_feat_test_df = extract_phenotype_feature(_tn_test_df)
    tn_pheno_feat_test_df.loc[:, 'LABEL'] = 0

    pheno_feat_train = pd.concat([tp_pheno_feat_train_df, tn_pheno_feat_train_df], ignore_index=True)
    pheno_feat_test = pd.concat([tp_pheno_feat_test_df, tn_pheno_feat_test_df], ignore_index=True)

    print('after filter the # of samples in training set: {}'.format(len(pheno_feat_train)))
    print('after filter the # of samples in testing set: {}'.format(len(pheno_feat_test)))

    pheno_feat = pd.concat([pheno_feat_train, pheno_feat_test], ignore_index=True)
    print('after filter the # of samples in whole dataset: {}'.format(len(pheno_feat)))
    pheno_symboles, pheno_colnames, pheno_values = separate_df(pheno_feat, ['SAMPLE_NUM', 'LABEL'])

    # imputation
    # missForest, ref: https://scikit-learn.org/stable/modules/impute.html
    # imputor = IterativeImputer(max_iter=200, random_state=0,
    #                            estimator=ExtraTreesRegressor(n_estimators=10, random_state=0))
    imputor = IterativeImputer(max_iter=100, random_state=0, sample_posterior=True)
    # imputor = IterativeImputer(max_iter=200, random_state=0,
    #                            estimator=RandomForestRegressor(n_estimators=20, random_state=0))
    pheno_imputed = imputor.fit_transform(pheno_values)

    data_root_dir = get_config()['data_dir']['data_root_dir']
    # save the imputor
    imputer_fname = os.path.join(data_root_dir, 'model_update/chr{}_ens_pheno-imputor.pkl'.format(chr))
    if os.path.exists(imputer_fname):
        os.remove(imputer_fname)
    joblib.dump(imputor, imputer_fname)

    # calculate BMI
    bmi = pheno_imputed[:, 2] / ((pheno_imputed[:, 1] / 100.0) ** 2)
    pheno_imputed_bmi = np.hstack((pheno_imputed, bmi.reshape(-1, 1)))

    # normalization
    scaler = StandardScaler()
    pheno_imputed_scaled = scaler.fit_transform(pheno_imputed_bmi)
    # save the scaler
    scaler_fname = os.path.join(data_root_dir, 'model_update/chr{}_ens_pheno-scaler.pkl'.format(chr))
    if os.path.exists(scaler_fname):
        os.remove(scaler_fname)
    joblib.dump(scaler, scaler_fname)

    pheno_trans_df = pd.DataFrame(data=pheno_imputed_scaled, columns=pheno_colnames + ['BMI'])
    # # calculate stata of non-normalization data
    # train_trans_df.describe().to_csv('stata.csv', sep='\t', index=False)
    pheno_trans_df[['SAMPLE_NUM', 'LABEL']] = pheno_symboles
    print('# of samples in phenotype set: {}'.format(pheno_trans_df.shape))

    pheno_train_df = pheno_trans_df[
        pheno_trans_df['SAMPLE_NUM'].isin(pheno_feat_train['SAMPLE_NUM'].values)
    ].copy().reset_index(drop=True)

    pheno_test_df = pheno_trans_df[
        pheno_trans_df['SAMPLE_NUM'].isin(pheno_feat_test['SAMPLE_NUM'].values)
    ].copy().reset_index(drop=True)

    print('after scaler, the number of samples in training set: {}'.format(len(pheno_train_df)))
    print('after scaler, the number of samples in testing set: {}'.format(len(pheno_test_df)))

    # load sequence feature
    tp_seq_features = extract_seq_feature(seq_type='tp', chr=chr)
    tn_seq_features = extract_seq_feature(seq_type='tn', chr=chr)
    seq_feature_df = pd.concat([tp_seq_features, tn_seq_features], ignore_index=True)
    print('# of train and test samples in seq: {}'.format(seq_feature_df.shape))
    seq_feat_samples, seq_feat_colnames, seq_feat_values = separate_df(seq_feature_df, ['Sample_id'])
    # normalize
    seq_scaler = StandardScaler()
    seq_feat_scaled = seq_scaler.fit_transform(seq_feat_values)
    seq_feat_scaled_df = pd.DataFrame(data=seq_feat_scaled, columns=seq_feat_colnames)
    seq_feat_scaled_df[['Sample_id']] = seq_feat_samples

    # save the seq_scaler
    seq_scaler_fname = os.path.join(data_root_dir, 'model_update/chr{}_ens_seq-scaler.pkl'.format(chr))
    if os.path.exists(seq_scaler_fname):
        os.remove(seq_scaler_fname)
    joblib.dump(seq_scaler, seq_scaler_fname)

    seq_train_df = seq_feat_scaled_df[
        seq_feat_scaled_df['Sample_id'].isin(pheno_feat_train['SAMPLE_NUM'].values)
    ].copy().reset_index(drop=True)
    seq_test_df = seq_feat_scaled_df[
        seq_feat_scaled_df['Sample_id'].isin(pheno_feat_test['SAMPLE_NUM'].values)
    ].copy().reset_index(drop=True)

    print('# of samples in seq for training: {}'.format(seq_train_df.shape))
    print('# of samples in seq for testing: {}'.format(seq_test_df.shape))

    # train_data
    train_df_ = pd.merge(pheno_train_df, seq_train_df, how='inner', left_on='SAMPLE_NUM', right_on='Sample_id')
    col_names = list(pheno_train_df.columns) + list(seq_train_df.columns)
    col_names_used = [x for x in col_names if x not in ['SAMPLE_NUM']]
    train_out_df = train_df_[col_names_used].copy()

    # test_data
    test_df_ = pd.merge(pheno_test_df, seq_test_df, how='inner', left_on='SAMPLE_NUM', right_on='Sample_id')
    test_col_names = list(pheno_test_df.columns) + list(seq_test_df.columns)
    test_col_names_used = [x for x in test_col_names if x not in ['SAMPLE_NUM']]
    test_out_df = test_df_[test_col_names_used].copy()

    if os.path.exists(train_feature_out_fname):
        os.remove(train_feature_out_fname)
    if os.path.exists(test_feature_out_fname):
        os.remove(test_feature_out_fname)
    train_out_df.to_csv(train_feature_out_fname, sep='\t', index=False)
    test_out_df.to_csv(test_feature_out_fname, sep='\t', index=False)
    return train_out_df, test_out_df


def read_features(reload=False, chr='21'):

    data_root_dir = get_config()['data_dir']['data_root_dir']
    train_feature_out_fname = os.path.join(data_root_dir,
                                           'model_update/results/chr{0}_normalized_ens_train.allfeature'.format(chr))
    test_feature_out_fname = os.path.join(data_root_dir,
                                           'model_update/results/chr{0}_normalized_ens_test.allfeature'.format(chr))
    if reload:
        if not os.path.exists(train_feature_out_fname):
            sys.exit('Error: {} not found.'.format(train_feature_out_fname))
        if not os.path.exists(test_feature_out_fname):
            sys.exit('Error: {} not found.'.format(test_feature_out_fname))
        train_df = pd.read_csv(train_feature_out_fname, sep='\t')
        test_df = pd.read_csv(test_feature_out_fname, sep='\t')
        return train_df, test_df

    _tp_train_df, _tp_test_df = get_train_test_samples(chr=chr, f_type='tp', reload=True)
    _tn_train_df, _tn_test_df = get_train_test_samples(chr=chr, f_type='tn', reload=True)

    print('the orignal # of samples in training set: {}'.format(len(_tp_train_df) + len(_tn_train_df)))
    print('the orignal # of samples in testing set: {}'.format(len(_tp_test_df) + len(_tn_test_df)))

    tp_pheno_feat_train_df = extract_phenotype_feature(_tp_train_df)
    tp_pheno_feat_train_df.loc[:, 'LABEL'] = 1
    tp_pheno_feat_test_df = extract_phenotype_feature(_tp_test_df)
    tp_pheno_feat_test_df.loc[:, 'LABEL'] = 1

    tn_pheno_feat_train_df = extract_phenotype_feature(_tn_train_df)
    tn_pheno_feat_train_df.loc[:, 'LABEL'] = 0
    tn_pheno_feat_test_df = extract_phenotype_feature(_tn_test_df)
    tn_pheno_feat_test_df.loc[:, 'LABEL'] = 0

    pheno_feat_train = pd.concat([tp_pheno_feat_train_df, tn_pheno_feat_train_df], ignore_index=True)
    pheno_feat_test = pd.concat([tp_pheno_feat_test_df, tn_pheno_feat_test_df], ignore_index=True)

    print('after filter the # of samples in training set: {}'.format(len(pheno_feat_train)))
    print('after filter the # of samples in testing set: {}'.format(len(pheno_feat_test)))

    train_symboles, train_colnames, train_values = separate_df(pheno_feat_train, ['SAMPLE_NUM', 'LABEL'])
    test_symboles, test_colnames, test_values = separate_df(pheno_feat_test, ['SAMPLE_NUM', 'LABEL'])

    # imputation
    # missForest, ref: https://scikit-learn.org/stable/modules/impute.html
    # imputor = IterativeImputer(max_iter=200, random_state=0,
    #                            estimator=ExtraTreesRegressor(n_estimators=10, random_state=0))
    imputor = IterativeImputer(max_iter=100, random_state=0, sample_posterior=True)
    # imputor = IterativeImputer(max_iter=200, random_state=0,
    #                            estimator=RandomForestRegressor(n_estimators=20, random_state=0))
    train_imputed = imputor.fit_transform(train_values)

    data_root_dir = get_config()['data_dir']['data_root_dir']
    # save the imputor
    imputer_fname = os.path.join(data_root_dir, 'model_update/chr{}_ens_pheno-imputor.pkl'.format(chr))
    if os.path.exists(imputer_fname):
        os.remove(imputer_fname)
    joblib.dump(imputor, imputer_fname)

    # calculate BMI
    bmi = train_imputed[:, 2] / ((train_imputed[:, 1] / 100.0) ** 2)
    train_imputed_bmi = np.hstack((train_imputed, bmi.reshape(-1, 1)))

    # normalization
    scaler = StandardScaler()
    train_imputed_scaled = scaler.fit_transform(train_imputed_bmi)
    # save the scaler
    scaler_fname = os.path.join(data_root_dir, 'model_update/chr{}_ens_pheno-scaler.pkl'.format(chr))
    if os.path.exists(scaler_fname):
        os.remove(scaler_fname)
    joblib.dump(scaler, scaler_fname)

    # test set
    test_imputed = imputor.transform(test_values)
    # calculate BMI
    test_bmi = test_imputed[:, 2] / ((test_imputed[:, 1] / 100.0) ** 2)
    test_imputed_bmi = np.hstack((test_imputed, test_bmi.reshape(-1, 1)))
    test_imputed_scaled = scaler.transform(test_imputed_bmi)

    train_trans_df = pd.DataFrame(data=train_imputed_scaled, columns=train_colnames+['BMI'])
    # # calculate stata of non-normalization data
    # train_trans_df.describe().to_csv('stata.csv', sep='\t', index=False)
    train_trans_df[['SAMPLE_NUM', 'LABEL']] = train_symboles
    print('# of training samples in phenotype set: {}'.format(train_trans_df.shape))

    test_trans_df = pd.DataFrame(data=test_imputed_scaled, columns=test_colnames+['BMI'])
    test_trans_df[['SAMPLE_NUM', 'LABEL']] = test_symboles
    print('# of testing samples in phenotype set: {}'.format(test_trans_df.shape))

    # load sequence feature
    tp_seq_features = extract_seq_feature(seq_type='tp', chr=chr)
    tn_seq_features = extract_seq_feature(seq_type='tn', chr=chr)
    seq_feature_df = pd.concat([tp_seq_features, tn_seq_features], ignore_index=True)
    print('# of train and test samples in seq: {}'.format(seq_feature_df.shape))

    seq_train_df = seq_feature_df[
        seq_feature_df['Sample_id'].isin(train_trans_df['SAMPLE_NUM'].values)
    ].copy().reset_index(drop=True)
    seq_test_df = seq_feature_df[
        seq_feature_df['Sample_id'].isin(test_trans_df['SAMPLE_NUM'].values)
    ].copy().reset_index(drop=True)

    print('# of samples in seq for training: {}'.format(seq_train_df.shape))
    print('# of samples in seq for testing: {}'.format(seq_test_df.shape))

    seq_train_samples, seq_train_colnames, seq_train_values = separate_df(seq_train_df, ['Sample_id'])
    seq_test_samples, seq_test_colnames, seq_test_values = separate_df(seq_test_df, ['Sample_id'])

    # normalize
    seq_scaler = StandardScaler()
    seq_train_scaled = seq_scaler.fit_transform(seq_train_values)
    seq_train_scaled_df = pd.DataFrame(data=seq_train_scaled, columns=seq_train_colnames)
    seq_train_scaled_df[['Sample_id']] = seq_train_samples

    # save the seq_scaler
    seq_scaler_fname = os.path.join(data_root_dir, 'model_update/chr{}_ens_seq-scaler.pkl'.format(chr))
    if os.path.exists(seq_scaler_fname):
        os.remove(seq_scaler_fname)
    joblib.dump(seq_scaler, seq_scaler_fname)

    seq_test_scaled = seq_scaler.transform(seq_test_values)
    seq_test_scaled_df = pd.DataFrame(data=seq_test_scaled, columns=seq_test_colnames)
    seq_test_scaled_df[['Sample_id']] = seq_test_samples

    # train_data
    train_df_ = pd.merge(train_trans_df, seq_train_scaled_df, how='inner', left_on='SAMPLE_NUM', right_on='Sample_id')
    col_names = list(train_trans_df.columns) + list(seq_train_scaled_df.columns)
    col_names_used = [x for x in col_names if x not in ['SAMPLE_NUM']]
    train_out_df = train_df_[col_names_used].copy()

    # test_data
    test_df_ = pd.merge(test_trans_df, seq_test_scaled_df, how='inner', left_on='SAMPLE_NUM', right_on='Sample_id')
    print('# number of test sample in final: {}'.format(test_df_.shape))
    test_col_names = list(test_trans_df.columns) + list(seq_test_scaled_df.columns)
    test_col_names_used = [x for x in test_col_names if x not in ['SAMPLE_NUM']]
    test_out_df = test_df_[test_col_names_used].copy()

    if os.path.exists(train_feature_out_fname):
        os.remove(train_feature_out_fname)
    if os.path.exists(test_feature_out_fname):
        os.remove(test_feature_out_fname)
    train_out_df.to_csv(train_feature_out_fname, sep='\t', index=False)
    test_out_df.to_csv(test_feature_out_fname, sep='\t', index=False)
    return train_out_df, test_out_df


def load_independent_data(seq_type='fp', chr='21', reload=False):
    """
    load fp, fn data for predicting
    :param seq_type:
    :param chr:
    :return:
    """
    data_root_dir = get_config()['data_dir']['data_root_dir']
    data_feature_out_fname = os.path.join(data_root_dir,
                                          'model_update/results',
                                          'chr{0}_normalized_ens_{1}.allfeature'.format(chr, seq_type))
    if reload:
        if not os.path.exists(data_feature_out_fname):
            sys.exit('Error: {} not found.'.format(data_feature_out_fname))
        return pd.read_csv(data_feature_out_fname, sep='\t')

    data_df = get_sample_info(reload=True, seq_type=seq_type, chr=chr)
    # get phenotype feature list
    data_pheno_feature_df = extract_phenotype_feature(data_df)
    data_symboles, data_colnames, data_values = separate_df(data_pheno_feature_df, ['SAMPLE_NUM'])

    pheno_imputor_fname = os.path.join(data_root_dir, 'model_update/chr{}_ens_pheno-imputor.pkl'.format(chr))
    if not os.path.exists(pheno_imputor_fname):
        sys.exit('Error: {} not found.'.format(pheno_imputor_fname))

    pheno_scaler_fname = os.path.join(data_root_dir, 'model_update/chr{}_ens_pheno-scaler.pkl'.format(chr))
    if not os.path.exists(pheno_scaler_fname):
        sys.exit('Error: {} not found.'.format(pheno_scaler_fname))

    seq_scaler_fname = os.path.join(data_root_dir, 'model_update/chr{}_ens_seq-scaler.pkl'.format(chr))
    if not os.path.exists(pheno_scaler_fname):
        sys.exit('Error: {} not found.'.format(pheno_scaler_fname))

    pheno_imputor = joblib.load(pheno_imputor_fname)
    pheno_scaler = joblib.load(pheno_scaler_fname)
    seq_scaler = joblib.load(seq_scaler_fname)

    data_imputed = pheno_imputor.transform(data_values)
    # calculate BMI
    data_bmi = data_imputed[:, 2] / ((data_imputed[:, 1] / 100.0) ** 2)
    data_imputed_bmi = np.hstack((data_imputed, data_bmi.reshape(-1, 1)))

    data_imputed_scaled = pheno_scaler.transform(data_imputed_bmi)
    data_trans_df = pd.DataFrame(data=data_imputed_scaled, columns=data_colnames + ['BMI'])
    data_trans_df[['SAMPLE_NUM']] = data_symboles

    data_seq_features = extract_seq_feature(seq_type=seq_type, chr=chr)
    data_seq_samples, data_seq_colnames, data_seq_values = separate_df(data_seq_features, ['Sample_id'])
    data_seq_scaled = seq_scaler.transform(data_seq_values)
    data_seq_scaled_df = pd.DataFrame(data=data_seq_scaled, columns=data_seq_colnames)
    data_seq_scaled_df[['Sample_id']] = data_seq_samples

    data_trans_final_df = pd.merge(data_trans_df, data_seq_scaled_df,
                                   how='inner', left_on='SAMPLE_NUM', right_on='Sample_id')
    col_names = list(data_trans_df.columns) + list(data_seq_scaled_df.columns)
    col_names_used = [x for x in col_names if x not in ['SAMPLE_NUM']]
    data_out_df = data_trans_final_df[col_names_used].copy()

    if os.path.exists(data_feature_out_fname):
        os.remove(data_feature_out_fname)
    data_out_df.to_csv(data_feature_out_fname, sep='\t', index=False)
    return data_out_df























