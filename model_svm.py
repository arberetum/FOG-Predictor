from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from ngram_features import get_ngrams_features
from fall_direction_features import get_all_fall_directions
from mml_features import get_mml_cui_CV_features, get_mml_cui_train_test_features
from prsf_features import get_prsf_features
from wordnet_features import get_verb_synset_features


rng = np.random.RandomState(11)


def grid_search_svm(train_texts, train_labels, param_grid):
    kf = KFold(n_splits=5)
    best_param_sets = []
    for i, (train_index, val_index) in enumerate(kf.split(train_texts)):
        these_train_texts = train_texts[train_index]
        these_train_labels = train_labels[train_index]
        these_val_texts = train_texts[val_index]
        these_val_labels = train_labels[val_index]
        # get feature sets
        train_ngram, val_ngram = get_ngrams_features(these_train_texts, these_val_texts)
        train_falldir = get_all_fall_directions(these_train_texts).to_numpy()
        val_falldir = get_all_fall_directions(these_val_texts).to_numpy()
        train_cui, val_cui = get_mml_cui_CV_features(train_index, val_index)
        train_prsf, val_prsf = get_prsf_features(these_train_texts, these_train_labels, these_val_texts)
        train_vb_syns, val_vb_syns, _ = get_verb_synset_features(these_train_texts, these_val_texts)
        train_feat = np.concatenate((train_ngram, train_falldir, train_cui, train_prsf, train_vb_syns), axis=1)
        val_feat = np.concatenate((val_ngram, val_falldir, val_cui, val_prsf, val_vb_syns), axis=1)
        print(f"Train Feature Matrix Shape: {train_feat.shape}")
        print(f"Validation Feature Matrix Shape: {val_feat.shape}")
        # scale features
        mms = MinMaxScaler(feature_range=(-1, 1))
        train_feat = mms.fit_transform(train_feat)
        val_feat = mms.transform(val_feat)
        # perform grid search
        svm = SVC(random_state=rng)
        svm_gs = GridSearchCV(svm, param_grid=param_grid, scoring='f1_micro', n_jobs=-1)
        svm_gs.fit(train_feat, these_train_labels)
        print(f"Fold {i}")
        print(f"Best Parameters: {svm_gs.best_params_}")
        print(f"Validation Accuracy: {svm_gs.score(val_feat, these_val_labels)}")
        best_param_sets.append(svm_gs.best_params_)
    return best_param_sets


if __name__ == '__main__':
    # load data
    train_df = pd.read_csv("./data/fallreports_2023-9-21_train.csv")
    train_texts = train_df['fall_description'].to_numpy(dtype=str)
    train_labels = train_df['fog_q_class'].to_numpy(dtype=int)
    test_df = pd.read_csv("./data/fallreports_2023-9-21_test.csv")
    test_texts = test_df['fall_description'].to_numpy(dtype=str)
    test_labels = test_df['fog_q_class'].to_numpy(dtype=int)

    # coarse grid search
    param_grid = dict()
    param_grid['C'] = [2.0**i for i in range(-5, 16, 2)]
    param_grid['class_weight'] = ['balanced', None]
    param_grid['kernel'] = ['linear', 'rbf']
    best_param_sets = grid_search_svm(train_texts, train_labels, param_grid)
    # results
    # Fold 0
    # Best Parameters: {'C': 8.0, 'class_weight': 'balanced', 'kernel': 'rbf'}
    # Validation Accuracy: 0.6833333333333333
    # Fold 1
    # Best Parameters: {'C': 8.0, 'class_weight': None, 'kernel': 'rbf'}
    # Validation Accuracy: 0.7166666666666667
    # Fold 2
    # Best Parameters: {'C': 8.0, 'class_weight': 'balanced', 'kernel': 'rbf'}
    # Validation Accuracy: 0.7
    # Fold 3
    # Best Parameters: {'C': 2.0, 'class_weight': 'balanced', 'kernel': 'rbf'}
    # Validation Accuracy: 0.6833333333333333
    # Fold 4
    # Best Parameters: {'C': 8.0, 'class_weight': 'balanced', 'kernel': 'rbf'}
    # Validation Accuracy: 0.8305084745762712

    # # fine grid search
    # param_grid = dict()
    # param_grid['C'] = np.arange(1, 16, 0.5)
    # param_grid['class_weight'] = ['balanced']
    # best_param_sets = grid_search_svm(train_texts, train_labels, param_grid)
    # results
    # Fold 0
    # Best Parameters: {'C': 4.0, 'class_weight': 'balanced'}
    # Validation Accuracy: 0.7333333333333333
    # Fold 1
    # Best Parameters: {'C': 7.5, 'class_weight': 'balanced'}
    # Validation Accuracy: 0.7166666666666667
    # Fold 2
    # Best Parameters: {'C': 9.5, 'class_weight': 'balanced'}
    # Validation Accuracy: 0.6833333333333333
    # Fold 3
    # Best Parameters: {'C': 12.0, 'class_weight': 'balanced'}
    # Validation Accuracy: 0.75
    # Fold 4
    # Best Parameters: {'C': 7.0, 'class_weight': 'balanced'}
    # Validation Accuracy: 0.8305084745762712

    # # get train and test features
    # train_ngram, test_ngram = get_ngrams_features(train_texts, test_texts)
    # train_falldir = get_all_fall_directions(train_texts).to_numpy()
    # test_falldir = get_all_fall_directions(test_texts).to_numpy()
    # train_cui, test_cui = get_mml_cui_train_test_features(len(train_texts), len(test_texts))
    # train_prsf, test_prsf = get_prsf_features(train_texts, train_labels, test_texts)
    # train_vb_syns, test_vb_syns, _ = get_verb_synset_features(train_texts, test_texts)
    # train_feat = np.concatenate((train_ngram, train_falldir, train_cui, train_prsf, train_vb_syns), axis=1)
    # test_feat = np.concatenate((test_ngram, test_falldir, test_cui, test_prsf, test_vb_syns), axis=1)
    # print(f"Train Feature Matrix Shape: {train_feat.shape}")
    # print(f"Test Feature Matrix Shape: {test_feat.shape}")
    #
    # # scale features
    # mms = MinMaxScaler(feature_range=(-1, 1))
    # train_feat = mms.fit_transform(train_feat)
    # test_feat = mms.transform(test_feat)
    #
    # # train model
    # best_params = {'C': 8, 'class_weight': 'balanced'}
    # svm = SVC(C=best_params['C'], class_weight=best_params['class_weight'], random_state=rng
    #           ).fit(train_feat, train_labels)
    #
    # # evaluate model
    # train_pred = svm.predict(train_feat)
    # acc_train = svm.score(train_feat, train_labels)
    # micro_f1_train = f1_score(train_labels, train_pred, average='micro')
    # macro_f1_train = f1_score(train_labels, train_pred, average='macro')
    # print(f"Train Accuracy: {acc_train}")
    # print(f"Train Micro F1: {micro_f1_train}")
    # print(f"Train Macro F1: {macro_f1_train}")
    # test_pred = svm.predict(test_feat)
    # acc_test = svm.score(test_feat, test_labels)
    # micro_f1_test = f1_score(test_labels, test_pred, average='micro')
    # macro_f1_test = f1_score(test_labels, test_pred, average='macro')
    # print(f"Test Accuracy: {acc_test}")
    # print(f"Test Micro F1: {micro_f1_test}")
    # print(f"Test Macro F1: {macro_f1_test}")
    # # results
    # # Train Feature Matrix Shape: (299, 13578)
    # # Test Feature Matrix Shape: (71, 13578)
    # # Train Accuracy: 1.0
    # # Train Micro F1: 1.0
    # # Train Macro F1: 1.0
    # # Test Accuracy: 0.7605633802816901
    # # Test Micro F1: 0.7605633802816902
    # # Test Macro F1: 0.7598009950248756