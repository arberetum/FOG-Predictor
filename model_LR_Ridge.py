from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from ngram_features import get_ngrams_features
from fall_direction_features import get_all_fall_directions
from mml_features import get_mml_cui_CV_features, get_mml_cui_train_test_features
from prsf_features import get_prsf_features
from wordnet_features import get_verb_synset_features


def grid_search(train_texts, train_labels, param_grid):
    kf = KFold(n_splits=5)
    best_Cs = []
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
        ss = StandardScaler()
        train_feat = ss.fit_transform(train_feat)
        val_feat = ss.transform(val_feat)
        # perform grid search
        lr = LogisticRegression(max_iter=3000)
        lr_gs = GridSearchCV(lr, param_grid=param_grid, scoring='f1_micro', n_jobs=-1)
        lr_gs.fit(train_feat, these_train_labels)
        print(f"Fold {i}")
        print(f"Best Parameters: {lr_gs.best_params_}")
        print(f"Validation Accuracy: {lr_gs.score(val_feat, these_val_labels)}")
        best_Cs.append(lr_gs.best_params_['C'])
    return np.mean(np.array(best_Cs))


if __name__ == '__main__':
    # load data
    train_df = pd.read_csv("./data/fallreports_2023-9-21_train.csv")
    train_texts = train_df['fall_description'].to_numpy(dtype=str)
    train_labels = train_df['fog_q_class'].to_numpy(dtype=int)
    test_df = pd.read_csv("./data/fallreports_2023-9-21_test.csv")
    test_texts = test_df['fall_description'].to_numpy(dtype=str)
    test_labels = test_df['fog_q_class'].to_numpy(dtype=int)

    # # coarse grid search
    # coarse_C_grid = [2.0**i for i in np.arange(start=-9, stop=16, step=2)]
    # coarse_param_grid = {'C': coarse_C_grid}
    # coarse_mean_C = grid_search(train_texts, train_labels, coarse_param_grid)
    # print(f"Mean C: {coarse_mean_C}")  # result = 128.028125, best C's = [0.0078125, 0.125, 128, 512, 0.0078125]

    # # fine grid search
    # fine_C_grid = [2.0**i for i in np.arange(start=-9, stop=0, step=0.5)]
    # fine_param_grid = {'C': fine_C_grid}
    # fine_mean_C = grid_search(train_texts, train_labels, fine_param_grid)
    # print(f"Mean C: {fine_mean_C}")  # result = 0.059472571625803015, best C's = [2^-7.5, 2^-3.5, 2^-6.5, 2^-2.5, 2^-6]

    # # another fine grid search
    # fine_C_grid_2 = np.arange(start=64, stop=200, step=10)
    # fine_param_grid_2 = {'C': fine_C_grid_2}
    # fine_mean_C_2 = grid_search(train_texts, train_labels, fine_param_grid_2)
    # print(f"Mean C: {fine_mean_C_2}")   # result = 112

    # get train and test features
    train_ngram, test_ngram = get_ngrams_features(train_texts, test_texts)
    train_falldir = get_all_fall_directions(train_texts).to_numpy()
    test_falldir = get_all_fall_directions(test_texts).to_numpy()
    train_cui, test_cui = get_mml_cui_train_test_features(len(train_texts), len(test_texts))
    train_prsf, test_prsf = get_prsf_features(train_texts, train_labels, test_texts)
    train_vb_syns, test_vb_syns, _ = get_verb_synset_features(train_texts, test_texts)
    train_feat = np.concatenate((train_ngram, train_falldir, train_cui, train_prsf, train_vb_syns), axis=1)
    test_feat = np.concatenate((test_ngram, test_falldir, test_cui, test_prsf, test_vb_syns), axis=1)
    print(f"Train Feature Matrix Shape: {train_feat.shape}")
    print(f"Test Feature Matrix Shape: {test_feat.shape}")

    # scale features
    ss = StandardScaler()
    train_feat = ss.fit_transform(train_feat)
    test_feat = ss.transform(test_feat)

    # train model
    lr = LogisticRegression(C=112, max_iter=3000).fit(train_feat, train_labels)

    # evaluate model
    train_pred = lr.predict(train_feat)
    acc_train = lr.score(train_feat, train_labels)
    micro_f1_train = f1_score(train_labels, train_pred, average='micro')
    macro_f1_train = f1_score(train_labels, train_pred, average='macro')
    print(f"Train Accuracy: {acc_train}")
    print(f"Train Micro F1: {micro_f1_train}")
    print(f"Train Macro F1: {macro_f1_train}")
    test_pred = lr.predict(test_feat)
    acc_test = lr.score(test_feat, test_labels)
    micro_f1_test = f1_score(test_labels, test_pred, average='micro')
    macro_f1_test = f1_score(test_labels, test_pred, average='macro')
    print(f"Test Accuracy: {acc_test}")
    print(f"Test Micro F1: {micro_f1_test}")
    print(f"Test Macro F1: {macro_f1_test}")
    # prints the following
    # Train Feature Matrix Shape: (299, 13578)
    # Test Feature Matrix Shape: (71, 13578)
    # Train Accuracy: 1.0
    # Train Micro F1: 1.0
    # Train Macro F1: 1.0
    # Test Accuracy: 0.7605633802816901
    # Test Micro F1: 0.7605633802816902
    # Test Macro F1: 0.7588411588411589