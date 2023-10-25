from sklearn.neighbors import KNeighborsClassifier
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


def random_search_knn(train_texts, train_labels, param_grid, n_iter):
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
        knn = KNeighborsClassifier()
        knn_rs = RandomizedSearchCV(knn, param_distributions=param_grid, scoring='f1_micro', n_jobs=-1, random_state=rng,
                                   n_iter=n_iter)
        knn_rs.fit(train_feat, these_train_labels)
        print(f"Fold {i}")
        print(f"Best Parameters: {knn_rs.best_params_}")
        print(f"Validation Accuracy: {knn_rs.score(val_feat, these_val_labels)}")
        best_param_sets.append(knn_rs.best_params_)
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
    # param_grid['n_neighbors'] = range(3, 18, 2)
    # param_grid['weights'] = ['uniform', 'distance']
    # param_grid['algorithm'] = ['auto', 'ball_tree', 'kd_tree', 'brute']
    # param_grid['leaf_size'] = range(10, 100, 10)
    # best_param_sets = random_search_knn(train_texts, train_labels, param_grid, n_iter)
    # results
    # Fold 0
    # Best Parameters: {'weights': 'distance', 'n_neighbors': 3, 'leaf_size': 20, 'algorithm': 'kd_tree'}
    # Validation Accuracy: 0.48333333333333334
    # Fold 1
    # Best Parameters: {'weights': 'distance', 'n_neighbors': 5, 'leaf_size': 70, 'algorithm': 'ball_tree'}
    # Validation Accuracy: 0.5333333333333333
    # Fold 2
    # Best Parameters: {'weights': 'distance', 'n_neighbors': 7, 'leaf_size': 10, 'algorithm': 'ball_tree'}
    # Validation Accuracy: 0.5166666666666667
    # Fold 3
    # Best Parameters: {'weights': 'distance', 'n_neighbors': 9, 'leaf_size': 30, 'algorithm': 'ball_tree'}
    # Validation Accuracy: 0.5333333333333333
    # Fold 4
    # Best Parameters: {'weights': 'distance', 'n_neighbors': 5, 'leaf_size': 40, 'algorithm': 'auto'}
    # Validation Accuracy: 0.5084745762711864

    # fine randomized hyperparameter search
    # n_iter = 100
    # param_grid = dict()
    # param_grid['n_neighbors'] = range(3, 18, 2)
    # param_grid['weights'] = ['distance']
    # param_grid['algorithm'] = ['ball_tree']
    # param_grid['leaf_size'] = range(2, 40, 2)
    # best_param_sets = random_search_knn(train_texts, train_labels, param_grid, n_iter)
    # results
    # Fold 0
    # Best Parameters: {'weights': 'distance', 'n_neighbors': 3, 'leaf_size': 16, 'algorithm': 'ball_tree'}
    # Validation Accuracy: 0.48333333333333334
    # Fold 1
    # Best Parameters: {'weights': 'distance', 'n_neighbors': 5, 'leaf_size': 30, 'algorithm': 'ball_tree'}
    # Validation Accuracy: 0.5333333333333333
    # Fold 2
    # Best Parameters: {'weights': 'distance', 'n_neighbors': 5, 'leaf_size': 2, 'algorithm': 'ball_tree'}
    # Validation Accuracy: 0.5166666666666667
    # Fold 3
    # Best Parameters: {'weights': 'distance', 'n_neighbors': 11, 'leaf_size': 18, 'algorithm': 'ball_tree'}
    # Validation Accuracy: 0.5166666666666667
    # Fold 4
    # Best Parameters: {'weights': 'distance', 'n_neighbors': 5, 'leaf_size': 6, 'algorithm': 'ball_tree'}
    # Validation Accuracy: 0.5084745762711864

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
    best_params = {'weights': 'distance', 'n_neighbors': 5, 'leaf_size': 14, 'algorithm': 'ball_tree'}
    knn = KNeighborsClassifier(weights=best_params['weights'], n_neighbors=best_params['n_neighbors'],
                               leaf_size=best_params['leaf_size'], algorithm=best_params['algorithm'],
                               n_jobs=-1).fit(train_feat, train_labels)

    # evaluate model
    train_pred = knn.predict(train_feat)
    acc_train = knn.score(train_feat, train_labels)
    micro_f1_train = f1_score(train_labels, train_pred, average='micro')
    macro_f1_train = f1_score(train_labels, train_pred, average='macro')
    print(f"Train Accuracy: {acc_train}")
    print(f"Train Micro F1: {micro_f1_train}")
    print(f"Train Macro F1: {macro_f1_train}")
    test_pred = knn.predict(test_feat)
    acc_test = knn.score(test_feat, test_labels)
    micro_f1_test = f1_score(test_labels, test_pred, average='micro')
    macro_f1_test = f1_score(test_labels, test_pred, average='macro')
    print(f"Test Accuracy: {acc_test}")
    print(f"Test Micro F1: {micro_f1_test}")
    print(f"Test Macro F1: {macro_f1_test}")
    # results
    # Train Accuracy: 1.0
    # Train Micro F1: 1.0
    # Train Macro F1: 1.0
    # Test Accuracy: 0.5492957746478874
    # Test Micro F1: 0.5492957746478874
    # Test Macro F1: 0.44313725490196076