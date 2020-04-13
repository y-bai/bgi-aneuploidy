#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: ens_model.py
    Description:
    
Created by YongBai on 2020/3/23 11:00 PM.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
# from mlxtend.classifier import StackingClassifier, StackingCVClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectFdr, SelectFpr, SelectFwe, SelectFromModel, SelectKBest, RFECV, RFE
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif


def _base_clfs(cls_weight=None):
    clf_svc = SVC(C=0.5, kernel="rbf", probability=True, random_state=10,
                  class_weight=cls_weight)
    clf_nusvc = NuSVC(kernel="rbf", nu=0.25, probability=True, random_state=10,  # nu=0.25
                      class_weight=cls_weight)
    clf_nb = GaussianNB()
    clf_rf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=5,
                                    max_features="auto", min_samples_leaf=1,
                                    min_samples_split=2, n_jobs=-1, random_state=5, class_weight=cls_weight)
    clf_xgb = XGBClassifier(n_estimators=100,
                            min_child_weight=1, gamma=0.0, colsample_bytree=0.8, subsample=0.7, reg_alpha=0.01,
                            max_depth=5, learning_rate=0.05, n_jobs=-1)

    clf_lr = LogisticRegression(C=0.5, solver='liblinear', class_weight=cls_weight)

    return [clf_svc, clf_nusvc, clf_nb, clf_rf, clf_xgb, clf_lr]


def stacking_model(f_est, cls_weight=None, kfold=5):
    # https://towardsdatascience.com/stacking-classifiers-for-higher-predictive-performance-566f963e4840
    if kfold > 0:
        from mlxtend.classifier import StackingCVClassifier
        sclf = StackingCVClassifier(classifiers=_base_clfs(cls_weight),
                                    shuffle=False,
                                    use_probas=True,
                                    cv=kfold,
                                    meta_classifier=f_est)
    else:
        base_ests = _base_clfs(cls_weight)
        base_names = [x.__class__.__name__ for x in base_ests]
        sclf = StackingClassifier(estimators=list(zip(base_names, base_ests)),
                                  final_estimator=f_est, stack_method='predict_proba', n_jobs=-1)
    return sclf


def ens_model_train(x_train, y_train, f_est, cls_weight=None):
    """
    this is using from sklearn.ensemble import StackingClassifier,
    which is not used in the current project
    :param x_train:
    :param y_train:
    :param f_est:
    :param cls_weight:
    :return:
    """
    # from sklearn.ensemble import StackingClassifier
    # if cls_weight is None:
    #     cls = np.unique(y_train)
    #     cls_wei = np.ones(len(cls)) * 0.5
    #     cls_weight = dict(zip(cls, cls_wei))
    # https://towardsdatascience.com/stacking-classifiers-for-higher-predictive-performance-566f963e4840
    clfs = [
        # ('knn', KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree', n_jobs=-1)),
        ('svc', SVC(C=50, degree=1, gamma="auto", kernel="rbf", probability=True, random_state=10,
                    class_weight=cls_weight)),
        ('nusvc', NuSVC(degree=1, kernel="rbf", nu=0.25, probability=True, random_state=10,
                        class_weight=cls_weight)),
        ('nb', GaussianNB()),
        # ('gp', GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=-1, random_state=10)),
        # ('etc', ExtraTreesClassifier(n_estimators=40, max_depth=5, max_features='auto', class_weight=cls_weight)),
        ('rf', RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=5,
                                     max_features="auto", min_samples_leaf=1,
                                     min_samples_split=2, n_jobs=-1, random_state=5, class_weight=cls_weight)),
        # ('gbm', GradientBoostingClassifier(n_estimators=100, max_depth=10,
        #                                    max_features='auto', learning_rate=0.01, random_state=10)),
        ('xgb', XGBClassifier(n_estimators=100,
                              min_child_weight=1, gamma=0.0, colsample_bytree=0.8, subsample=0.7, reg_alpha=0.01,
                              max_depth=5, learning_rate=0.05, n_jobs=-1))
    ]

    # f_est = LogisticRegression(class_weight=cls_weight)

    # f_est = ExtraTreesClassifier(
    #     n_estimators=100, max_depth=10, max_features='auto', class_weight=cls_weight
    # )

    f_clf = StackingClassifier(estimators=clfs, final_estimator=f_est, stack_method='predict_proba', n_jobs=-1)
    # f_clf = StackingClassifier(estimators=clfs,
    #                            final_estimator=ExtraTreesClassifier(
    #                                n_estimators=100, max_depth=10, max_features='auto', class_weight=cls_weight
    #                            ),
    #                            ,
    #                            n_jobs=-1)
    f_clf.fit(x_train, y_train)
    return f_clf


def feature_selection(x, y, sel_method='estimator', k=None,  estimator=None, score_func=chi2):
    """
    :param x:
    :param y:
    :param k:
    :param sel_method: kbest, fdr, fpr, fwe, estimator, rfecv
    :param estimator:
    :param score_func:
    :return:
    """

    if sel_method == 'kbest':
        assert k is not None
        selector = SelectKBest(score_func, k)
    elif sel_method == 'fdr':
        selector = SelectFdr(score_func, alpha=0.05)
    elif sel_method == 'fpr':
        selector = SelectFpr(score_func, alpha=0.05)
    elif sel_method == 'fwe':
        selector = SelectFwe(score_func, alpha=0.05)
    elif sel_method == 'estimator':
        assert estimator is not None
        if k is None:
            selector = SelectFromModel(estimator=estimator)
        else:
            selector = SelectFromModel(estimator=estimator, max_features=k, threshold=-np.inf)
    elif sel_method == 'rfecv':
        assert estimator is not None
        # selector = RFECV(estimator, step=1, cv=5)
        selector = RFE(estimator, step=1)
    else:
        raise Exception('unknown input parameters.')

    assert selector is not None
    x_new = selector.fit_transform(x, y)
    return selector, x_new, y



