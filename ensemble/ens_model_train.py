#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: ens_model_train.py
    Description:
    
Created by YongBai on 2020/3/24 10:39 AM.
"""
import os
import numpy as np
import pandas as pd
from scipy import interp
import joblib
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import plot_roc_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import ExtraTreesClassifier
from .data_load import read_features
from .ens_model import *
import sys
sys.path.append('..')
from utils import get_config


def get_class_weight(y_train):
    cls = np.unique(y_train)
    cls_weight = compute_class_weight('balanced', cls, y_train)
    class_weight_dict = dict(zip(cls, cls_weight))
    return class_weight_dict


def cv_feature_selection(chr='21', kfold=5):
    """
    this function not used at this moment, as we don't need feature selection
    :param chr:
    :param kfold:
    :return:
    """
    data_df, _ = read_features(reload=True, chr=chr)

    # data_df = pd.concat([train_out_df, test_out_df], ignore_index=True)

    y_data_arr = data_df['LABEL'].values
    x_data_df = data_df.drop(columns=['Sample_id', 'LABEL'])
    raw_feat_name = list(x_data_df.columns)
    x_data_arr = x_data_df.values

    est = SVC(kernel="linear")
    kfold_score = []
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=0)
    for train_index, val_index in skf.split(x_data_arr, y_data_arr):
        X_train, X_val = x_data_arr[train_index], x_data_arr[val_index]
        y_train, y_val = y_data_arr[train_index], y_data_arr[val_index]

        selector, x_train_new, y_train_new = feature_selection(
            X_train, y_train,
            k=8,
            estimator=est, #ExtraTreesClassifier(n_estimators=100, max_depth=5, max_features='auto'),
            sel_method='rfecv')
        supp = selector.get_support()
        print('selected features: {}'.format(list(np.array(raw_feat_name)[supp])))
        # print(selector.estimator_.feature_importances_)
        # print(selector.estimator_.coef_)
        # print(selector.threshold_)
        clf = LogisticRegression(random_state=0,
                                 solver='liblinear',
                                 max_iter=200,
                                 class_weight=get_class_weight(y_train_new)).fit(x_train_new, y_train_new)
        kfold_score.append(clf.score(X_val[:, supp], y_val))

    kfold_score_arr = np.array(kfold_score)
    print('5-fold sorce: {}'.format(kfold_score_arr))
    print('avg score: {}'.format(np.mean(kfold_score_arr)))


def cv_ens_model_train_run(chr='21', kfold=5):

    # loading the training dataset
    train_out_df, _ = read_features(reload=True, chr=chr)

    # use_col = ['AGE', 'GESTATIONAL_WEEKS_D', 'SEQ_F_1', 'SEQ_F_2', 'SEQ_F_3', 'SEQ_F_4', 'SEQ_F_5', 'SEQ_F_6', 'Sample_id', 'LABEL']
    # train_out_df = train_out_df_t[use_col].copy()

    y_data_arr = train_out_df['LABEL'].values
    x_data_arr = train_out_df.drop(columns=['Sample_id', 'LABEL']).values

    cls_w = get_class_weight(y_data_arr)

    # f_est = LogisticRegression(n_jobs=-1, class_weight=cls_w)  # solver='liblinear',
    # f_est_params_to_tune = {
    #     'meta_classifier__C': [0.01, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 5.0]  # best C = 0.5
    # }
    f_est_params_to_tune = {
        # 'svc__C': [0.1, 0.2, 0.5, 1, 2, 5],
        # 'nusvc__nu': [0.1, 0.2, 0.25, 0.5, 0.75],
        # 'randomforestclassifier__n_estimators': range(80, 120, 10),
        # 'randomforestclassifier__max_depth': range(2, 11, 2),
        # 'randomforestclassifier__min_samples_split': range(2, 5),
        # 'xgbclassifier__n_estimators': range(80, 120, 10),
        # 'xgbclassifier__max_depth': range(2, 11, 2),
        # 'xgbclassifier__min_child_weight': range(1, 4),
        # 'xgbclassifier__gamma': [i / 10.0 for i in range(0, 5)],
        # 'xgbclassifier__learning_rate': [.2, .1, .05, .01],
        # 'xgbclassifier__reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
        'meta_classifier__n_estimators': range(80, 120, 10),
        'meta_classifier__max_depth': range(2, 11, 2),
        'meta_classifier__min_samples_split': range(2, 5),
    }
    # 'meta_classifier__max_depth': 6, 'meta_classifier__min_samples_split': 3, 'meta_classifier__n_estimators': 110}
    # or 'meta_classifier__max_depth': 6, 'meta_classifier__min_samples_split': 4, 'meta_classifier__n_estimators': 100}
    # 0.9620885514450702
    f_est = ExtraTreesClassifier(class_weight=cls_w,
                                 max_depth=6, min_samples_split=3, n_estimators=110,
                                 min_samples_leaf=1, n_jobs=-1)
    sclf = stacking_model(f_est, cls_weight=cls_w, kfold=kfold)

    sclf_cv = GridSearchCV(estimator=sclf,
                        param_grid=f_est_params_to_tune,
                        cv=kfold,
                        scoring="roc_auc",
                        verbose=10,
                        n_jobs=10)
    sclf_cv.fit(x_data_arr, y_data_arr)
    print(sclf_cv.best_params_)
    re_df = pd.DataFrame(sclf_cv.cv_results_)
    re_df.to_csv('re_df.csv', index=False, sep='\t')


def ens_model_final_train(chr='21', kfold=5):

    train_out_df, _ = read_features(reload=True, chr=chr)

    # use_col = ['SEQ_F_0', 'SEQ_F_1', 'SEQ_F_2', 'SEQ_F_3', 'SEQ_F_4', 'SEQ_F_5', 'SEQ_F_6',
    #  'SEQ_F_7', 'Sample_id', 'LABEL']
    # train_out_df = train_out_t_df[use_col].copy()

    y_data_arr = train_out_df['LABEL'].values
    x_data_arr = train_out_df.drop(columns=['Sample_id', 'LABEL']).values

    cls_w = get_class_weight(y_data_arr)

    f_est = ExtraTreesClassifier(class_weight=cls_w,
                                 max_depth=6, min_samples_split=3, n_estimators=110,
                                 min_samples_leaf=1, n_jobs=-1)
    sclf = stacking_model(f_est, cls_weight=cls_w, kfold=0)

    tprs = []
    aucs = []
    opt_thresholds = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=0)
    for i, (train_index, val_index) in enumerate(skf.split(x_data_arr, y_data_arr)):
        logging.info('training on {} fold cv...'.format(i))
        X_train, X_val = x_data_arr[train_index], x_data_arr[val_index]
        y_train, y_val = y_data_arr[train_index], y_data_arr[val_index]

        f_clf = sclf.fit(X_train, y_train)
        y_pred_proba = f_clf.predict_proba(X_val)
        fpr_, tpr_, thresholds_ = roc_curve(y_val, y_pred_proba[:, 1])

        optimal_idx = np.argmax(tpr_ - fpr_)
        optimal_threshold = thresholds_[optimal_idx]
        opt_thresholds.append(optimal_threshold)

        viz = plot_roc_curve(f_clf, X_val, y_val, name='{}-Fold'.format(i+1),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic")
    ax.legend()
    plt.savefig('ens_roc.png', dpi=300)

    # optimal threshods: [0.5594350756023522, 0.3857846509217087, 0.49505949768890123, 0.5041115716755037, 0.46716558896678595]
    # mean of optimal threshold: 0.4823112769710504
    mean_opt_thre = np.mean(opt_thresholds)
    print('optimal threshods: {}'.format(opt_thresholds))
    print('mean of optimal threshold: {}'.format(mean_opt_thre))

    ens_model_out_fname = os.path.join(get_config()['data_dir']['data_root_dir'],
                                       'model_update/ensemble_model_chr{}.pkl'.format(chr))
    if os.path.exists(ens_model_out_fname):
        os.remove(ens_model_out_fname)
    # model train on the whole train dataset
    logging.info('training on whole training set...')
    final_clf = sclf.fit(x_data_arr, y_data_arr)
    logging.info('saved the model trained on whole training set... {}'.format(ens_model_out_fname))
    joblib.dump(final_clf, ens_model_out_fname)


def ens_model_predict(x_in_arr):

    ens_model_fname = os.path.join(get_config()['data_dir']['data_root_dir'], 'model', 'ensemble_model_train.pkl')
    if not os.path.exists(ens_model_fname):
        sys.exit('Error: {} not found.'.format(ens_model_fname))
    ens_model = joblib.load(ens_model_fname)









