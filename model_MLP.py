from sklearn.neural_network import MLPClassifier
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
import matplotlib.pyplot as plt


rng = np.random.RandomState(11)


def grid_search_mlp(train_texts, train_labels, param_grid):
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
        mlp = MLPClassifier(random_state=rng)
        mlp_gs = GridSearchCV(mlp, param_grid=param_grid, scoring='f1_micro', n_jobs=-3)
        mlp_gs.fit(train_feat, these_train_labels)
        print(f"Fold {i}")
        print(f"Best Parameters: {mlp_gs.best_params_}")
        print(f"Validation Accuracy: {mlp_gs.score(val_feat, these_val_labels)}")
        best_param_sets.append(mlp_gs.best_params_)
    return best_param_sets


if __name__ == '__main__':
    # load data
    train_df = pd.read_csv("./data/fallreports_2023-9-21_train.csv")
    train_texts = train_df['fall_description'].to_numpy(dtype=str)
    train_labels = train_df['fog_q_class'].to_numpy(dtype=int)
    test_df = pd.read_csv("./data/fallreports_2023-9-21_test.csv")
    test_texts = test_df['fall_description'].to_numpy(dtype=str)
    test_labels = test_df['fog_q_class'].to_numpy(dtype=int)

    # # coarse grid search
    # param_grid = dict()
    # param_grid['hidden_layer_sizes'] = [(3000,), (1000,), (100,), (3000, 3000), (1000, 1000),
    #                                     (100, 100), (1000, 1000, 1000), (100, 100, 100)] #[(9000,), (6000,), (3000,), (1000,), (100,), (3000, 3000), (1000, 1000),
    #                                     # (100, 100), (1000, 1000, 1000), (100, 100, 100)]
    # param_grid['alpha'] = [10.0**i for i in range(-6, -1)]
    # # param_grid['learning_rate'] = ['constant', 'invscaling', 'adaptive']
    # best_param_sets = grid_search_mlp(train_texts, train_labels, param_grid)
    # results
    # Fold 0
    # Best Parameters: {'alpha': 0.001, 'hidden_layer_sizes': (1000, 1000)}
    # Validation Accuracy: 0.6833333333333333
    # Fold 1 - some did not converge
    # Best Parameters: {'alpha': 1e-06, 'hidden_layer_sizes': (3000, 3000)}
    # Validation Accuracy: 0.7
    # Fold 2 - some did not converge
    # Best Parameters: {'alpha': 0.01, 'hidden_layer_sizes': (3000, 3000)}
    # Validation Accuracy: 0.6833333333333333
    # Fold 3 - some did not converge
    # Best Parameters: {'alpha': 0.0001, 'hidden_layer_sizes': (1000, 1000, 1000)}
    # Validation Accuracy: 0.7
    # Fold 4 - some did not converge
    # Best Parameters: {'alpha': 0.01, 'hidden_layer_sizes': (3000,)}
    # Validation Accuracy: 0.711864406779661

    # # second grid search
    # param_grid = dict()
    # param_grid['hidden_layer_sizes'] = [(3000, 3000), (1000, 1000), (100, 100), (1000, 1000, 1000), (100, 100, 100)]
    # param_grid['alpha'] = [10.0 ** -4]
    # param_grid['learning_rate'] = ['constant', 'invscaling', 'adaptive']
    # param_grid['max_iter'] = [500]
    # param_grid['early_stopping'] = [True]
    # best_param_sets = grid_search_mlp(train_texts, train_labels, param_grid)
    # results
    # Fold 0
    # Best Parameters: {'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (3000, 3000), 'learning_rate': 'constant', 'max_iter': 500}
    # Validation Accuracy: 0.6833333333333333
    # Fold 1
    # Best Parameters: {'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (3000, 3000), 'learning_rate': 'constant', 'max_iter': 500}
    # Validation Accuracy: 0.55
    # Fold 2
    # Best Parameters: {'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (1000, 1000, 1000), 'learning_rate': 'constant', 'max_iter': 500}
    # Validation Accuracy: 0.6833333333333333
    # Fold 3
    # Best Parameters: {'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (100, 100), 'learning_rate': 'constant', 'max_iter': 500}
    # Validation Accuracy: 0.7
    # Fold 4
    # Best Parameters: {'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (1000, 1000, 1000), 'learning_rate': 'constant', 'max_iter': 500}
    # Validation Accuracy: 0.711864406779661

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
    mlp = MLPClassifier(hidden_layer_sizes=(1000, 1000, 1000), alpha=10.0**-4, learning_rate='constant', max_iter=500,
                        random_state=rng, early_stopping=True).fit(train_feat, train_labels)
    plt.plot(list(mlp.loss_curve_))
    plt.plot(list(mlp.validation_scores_))
    plt.show()

    # evaluate model
    train_pred = mlp.predict(train_feat)
    acc_train = mlp.score(train_feat, train_labels)
    micro_f1_train = f1_score(train_labels, train_pred, average='micro')
    macro_f1_train = f1_score(train_labels, train_pred, average='macro')
    print(f"Train Accuracy: {acc_train}")
    print(f"Train Micro F1: {micro_f1_train}")
    print(f"Train Macro F1: {macro_f1_train}")
    test_pred = mlp.predict(test_feat)
    acc_test = mlp.score(test_feat, test_labels)
    micro_f1_test = f1_score(test_labels, test_pred, average='micro')
    macro_f1_test = f1_score(test_labels, test_pred, average='macro')
    print(f"Test Accuracy: {acc_test}")
    print(f"Test Micro F1: {micro_f1_test}")
    print(f"Test Macro F1: {macro_f1_test}")
    # results
    # Train Feature Matrix Shape: (299, 13578)
    # Test Feature Matrix Shape: (71, 13578)
    # Train Accuracy: 0.9732441471571907
    # Train Micro F1: 0.9732441471571907
    # Train Macro F1: 0.973193473193473
    # Test Accuracy: 0.6338028169014085
    # Test Micro F1: 0.6338028169014085
    # Test Macro F1: 0.633147853736089