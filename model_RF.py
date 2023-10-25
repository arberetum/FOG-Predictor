from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from ngram_features import get_ngrams_features
from fall_direction_features import get_all_fall_directions
from mml_features import get_mml_cui_CV_features, get_mml_cui_train_test_features
from prsf_features import get_prsf_features
from wordnet_features import get_verb_synset_features


rng = np.random.RandomState(11)


def random_search_rf(train_texts, train_labels, param_grid, n_iter):
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
        ss = StandardScaler()
        train_feat = ss.fit_transform(train_feat)
        val_feat = ss.transform(val_feat)
        # perform grid search
        rf = RandomForestClassifier(random_state=rng)
        rf_gs = RandomizedSearchCV(rf, param_distributions=param_grid, scoring='f1_micro', n_jobs=-1, random_state=rng,
                                   n_iter=n_iter)
        rf_gs.fit(train_feat, these_train_labels)
        print(f"Fold {i}")
        print(f"Best Parameters: {rf_gs.best_params_}")
        print(f"Validation Accuracy: {rf_gs.score(val_feat, these_val_labels)}")
        best_param_sets.append(rf_gs.best_params_)
    return best_param_sets


if __name__ == '__main__':
    # load data
    train_df = pd.read_csv("./data/fallreports_2023-9-21_train.csv")
    train_texts = train_df['fall_description'].to_numpy(dtype=str)
    train_labels = train_df['fog_q_class'].to_numpy(dtype=int)
    test_df = pd.read_csv("./data/fallreports_2023-9-21_test.csv")
    test_texts = test_df['fall_description'].to_numpy(dtype=str)
    test_labels = test_df['fog_q_class'].to_numpy(dtype=int)

    # # coarse randomized hyperparameter search
    # n_iter = 100
    # param_grid = dict()
    # param_grid['criterion'] = ['gini', 'entropy', 'log_loss']
    # param_grid['max_depth'] = [None] + list(range(10, 201, 20))
    # param_grid['min_samples_leaf'] = range(1, 11)
    # param_grid['max_features'] = ['sqrt', 'log2', None]
    # best_param_sets = random_search_rf(train_texts, train_labels, param_grid, n_iter)
    # results
    # Fold 0
    # Best Parameters: {'min_samples_leaf': 3, 'max_features': None, 'max_depth': 70, 'criterion': 'log_loss'}
    # Validation Accuracy: 0.7333333333333333
    # Fold 1
    # Best Parameters: {'min_samples_leaf': 1, 'max_features': None, 'max_depth': 70, 'criterion': 'gini'}
    # Validation Accuracy: 0.6833333333333333
    # Fold 2
    # Best Parameters: {'min_samples_leaf': 1, 'max_features': None, 'max_depth': 30, 'criterion': 'entropy'}
    # Validation Accuracy: 0.7833333333333333
    # Fold 3
    # Best Parameters: {'min_samples_leaf': 1, 'max_features': None, 'max_depth': 10, 'criterion': 'gini'}
    # Validation Accuracy: 0.7833333333333333
    # Fold 4
    # Best Parameters: {'min_samples_leaf': 2, 'max_features': None, 'max_depth': 170, 'criterion': 'log_loss'}
    # Validation Accuracy: 0.7966101694915254

    # # fine randomized hyperparameter search
    # n_iter = 100
    # param_grid = dict()
    # param_grid['criterion'] = ['gini', 'entropy', 'log_loss']
    # param_grid['max_depth'] = [None] + list(range(10, 201, 10))
    # param_grid['min_samples_leaf'] = [1, 2, 3]
    # param_grid['max_features'] = [None]
    # best_param_sets = random_search_rf(train_texts, train_labels, param_grid, n_iter)
    # results
    # Fold 0
    # Best Parameters: {'min_samples_leaf': 3, 'max_features': None, 'max_depth': None, 'criterion': 'log_loss'}
    # Validation Accuracy: 0.7333333333333333
    # Fold 1
    # Best Parameters: {'min_samples_leaf': 2, 'max_features': None, 'max_depth': 10, 'criterion': 'gini'}
    # Validation Accuracy: 0.7333333333333333
    # Fold 2
    # Best Parameters: {'min_samples_leaf': 3, 'max_features': None, 'max_depth': 130, 'criterion': 'gini'}
    # Validation Accuracy: 0.7166666666666667
    # Fold 3
    # Best Parameters: {'min_samples_leaf': 1, 'max_features': None, 'max_depth': 10, 'criterion': 'gini'}
    # Validation Accuracy: 0.7
    # Fold 4
    # Best Parameters: {'min_samples_leaf': 2, 'max_features': None, 'max_depth': 80, 'criterion': 'gini'}
    # Validation Accuracy: 0.7627118644067796

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
    best_params = {'min_samples_leaf': 2, 'max_features': None, 'max_depth': 60, 'criterion': 'gini'}
    rf = RandomForestClassifier(min_samples_leaf=best_params['min_samples_leaf'], max_features=best_params['max_features'],
                                max_depth=best_params['max_depth'], criterion=best_params['criterion'],
                                random_state=rng, n_jobs=-1).fit(train_feat, train_labels)

    # evaluate model
    train_pred = rf.predict(train_feat)
    acc_train = rf.score(train_feat, train_labels)
    micro_f1_train = f1_score(train_labels, train_pred, average='micro')
    macro_f1_train = f1_score(train_labels, train_pred, average='macro')
    print(f"Train Accuracy: {acc_train}")
    print(f"Train Micro F1: {micro_f1_train}")
    print(f"Train Macro F1: {macro_f1_train}")
    test_pred = rf.predict(test_feat)
    acc_test = rf.score(test_feat, test_labels)
    micro_f1_test = f1_score(test_labels, test_pred, average='micro')
    macro_f1_test = f1_score(test_labels, test_pred, average='macro')
    print(f"Test Accuracy: {acc_test}")
    print(f"Test Micro F1: {micro_f1_test}")
    print(f"Test Macro F1: {macro_f1_test}")
    # results
    # Train Feature Matrix Shape: (299, 13578)
    # Test Feature Matrix Shape: (71, 13578)
    # Train Accuracy: 0.9966555183946488
    # Train Micro F1: 0.9966555183946488
    # Train Macro F1: 0.9966501226794537
    # Test Accuracy: 0.6901408450704225
    # Test Micro F1: 0.6901408450704225
    # Test Macro F1: 0.6895866454689984