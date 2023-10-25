from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
import numpy as np
import pandas as pd
from ngram_features import get_ngrams_features
from fall_direction_features import get_all_fall_directions
from mml_features import get_mml_cui_CV_features, get_mml_cui_train_test_features
from prsf_features import get_prsf_features
from wordnet_features import get_verb_synset_features


rng = np.random.RandomState(11)


if __name__ == '__main__':
    # load data
    train_df = pd.read_csv("./data/fallreports_2023-9-21_train.csv")
    train_texts = train_df['fall_description'].to_numpy(dtype=str)
    train_labels = train_df['fog_q_class'].to_numpy(dtype=int)
    test_df = pd.read_csv("./data/fallreports_2023-9-21_test.csv")
    test_texts = test_df['fall_description'].to_numpy(dtype=str)
    test_labels = test_df['fog_q_class'].to_numpy(dtype=int)

    # get train and test features
    train_ngram, test_ngram = get_ngrams_features(train_texts, test_texts)
    train_falldir = get_all_fall_directions(train_texts).to_numpy()
    test_falldir = get_all_fall_directions(test_texts).to_numpy()
    train_cui, test_cui = get_mml_cui_train_test_features(len(train_texts), len(test_texts))
    train_prsf, test_prsf = get_prsf_features(train_texts, train_labels, test_texts)
    train_vb_syns, test_vb_syns, _ = get_verb_synset_features(train_texts, test_texts)

    # created ablated feature sets
    train_feat_grps = dict()
    train_feat_grps['n-grams'] = np.concatenate((train_falldir, train_cui, train_prsf, train_vb_syns), axis=1)
    train_feat_grps['fall directions'] = np.concatenate((train_ngram, train_cui, train_prsf, train_vb_syns), axis=1)
    train_feat_grps['concept IDs'] = np.concatenate((train_ngram, train_falldir, train_prsf, train_vb_syns), axis=1)
    train_feat_grps['P-RSF metrics'] = np.concatenate((train_ngram, train_falldir, train_cui, train_vb_syns), axis=1)
    train_feat_grps['verb synsets'] = np.concatenate((train_ngram, train_falldir, train_cui, train_prsf), axis=1)
    test_feat_grps = dict()
    test_feat_grps['n-grams'] = np.concatenate((test_falldir, test_cui, test_prsf, test_vb_syns), axis=1)
    test_feat_grps['fall directions'] = np.concatenate((test_ngram, test_cui, test_prsf, test_vb_syns), axis=1)
    test_feat_grps['concept IDs'] = np.concatenate((test_ngram, test_falldir, test_prsf, test_vb_syns), axis=1)
    test_feat_grps['P-RSF metrics'] = np.concatenate((test_ngram, test_falldir, test_cui, test_vb_syns), axis=1)
    test_feat_grps['verb synsets'] = np.concatenate((test_ngram, test_falldir, test_cui, test_prsf), axis=1)

    # loop over different feature sets
    for set_removed in train_feat_grps.keys():
        train_feat = train_feat_grps[set_removed]
        test_feat = test_feat_grps[set_removed]

        print(f"Feature Group Removed: {set_removed}")
        print(f"Train Feature Matrix Shape: {train_feat.shape}")
        print(f"Test Feature Matrix Shape: {test_feat.shape}")

        # scale features
        ss = StandardScaler()
        train_feat = ss.fit_transform(train_feat)
        test_feat = ss.transform(test_feat)

        # randomly hold out 10% of the training data for training the final classifier
        holdout_inds = np.random.choice(np.arange(0, len(train_texts)), size=(int(np.floor(0.1*len(train_texts))),),
                                        replace=False)
        kept_inds = [ind for ind in np.arange(0, len(train_texts)) if ind not in holdout_inds]
        base_train_feat = train_feat[kept_inds, :]
        base_train_labels = train_labels[kept_inds]
        stack_train_feat = train_feat[holdout_inds, :]
        stack_train_labels = train_labels[holdout_inds]

        # train base models
        nb = GaussianNB().fit(base_train_feat, base_train_labels)
        svm = SVC(C=8, class_weight='balanced', random_state=rng).fit(base_train_feat, base_train_labels)
        rf = RandomForestClassifier(min_samples_leaf=2, max_features=None, max_depth=60, criterion='gini',
                                    random_state=rng).fit(base_train_feat, base_train_labels)
        knn = KNeighborsClassifier(weights='distance', n_neighbors=5, leaf_size=14, algorithm='ball_tree'
                                   ).fit(base_train_feat, base_train_labels)
        mlp = MLPClassifier(hidden_layer_sizes=(1000, 1000, 1000), alpha=10.0**-4, learning_rate='constant', max_iter=500,
                            random_state=rng, early_stopping=True).fit(base_train_feat, base_train_labels)

        # train stacked classifier
        estimators = [('nb', nb), ('svm', svm), ('rf', rf), ('knn', knn), ('mlp', mlp)]
        sc = StackingClassifier(estimators=estimators, cv='prefit').fit(stack_train_feat, stack_train_labels)

        # evaluate model
        test_pred = sc.predict(test_feat)
        acc_test = sc.score(test_feat, test_labels)
        micro_f1_test = f1_score(test_labels, test_pred, average='micro')
        macro_f1_test = f1_score(test_labels, test_pred, average='macro')
        print(f"Test Accuracy: {acc_test}")
        print(f"Test Micro F1: {micro_f1_test}")
        print(f"Test Macro F1: {macro_f1_test}")

