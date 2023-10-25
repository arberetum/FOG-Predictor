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
import matplotlib.pyplot as plt


rng = np.random.RandomState(11)


if __name__ == '__main__':
    # load data
    train_df = pd.read_csv("./data/fallreports_2023-9-21_train.csv")
    train_texts_full = train_df['fall_description'].to_numpy(dtype=str)
    train_labels_full = train_df['fog_q_class'].to_numpy(dtype=int)
    test_df = pd.read_csv("./data/fallreports_2023-9-21_test.csv")
    test_texts = test_df['fall_description'].to_numpy(dtype=str)
    test_labels = test_df['fog_q_class'].to_numpy(dtype=int)

    micro_f1s = []
    for train_fraction in np.arange(0.2, 1.1, 0.1):
        # train_inds = np.random.choice(np.arange(0, len(train_texts_full)),
        #                               size=(int(np.floor(train_fraction * len(train_texts_full))),), replace=False)
        train_texts = train_texts_full[range(0, int(np.floor(train_fraction * len(train_texts_full))))]
        train_labels = train_labels_full[range(0, int(np.floor(train_fraction * len(train_texts_full))))]

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

        # randomly hold out 10% of the training data for training the final classifier
        holdout_inds = np.random.choice(np.arange(0, len(train_texts)), size=(int(np.floor(0.1*train_feat.shape[0])),),
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
        micro_f1s.append(micro_f1_test)

    x = np.arange(0.2, 1.1, 0.1) * len(train_texts)
    plt.plot(x, micro_f1s, 'bo')
    log_fit = np.polyfit(np.log(x), micro_f1s, deg=1)
    x_fine = np.arange(10, len(train_texts) + 200, 1)
    y_fit = log_fit[0]*np.log(x_fine) + log_fit[1]
    plt.plot(x_fine, y_fit, 'k-')
    plt.xlabel("Training Set Size", fontsize='large')
    plt.ylabel("Micro-averaged F1 Score", fontsize='large')
    ax = plt.gca()
    ax.text(125, 0.6, f"y={log_fit[0]:.3f}ln(x){log_fit[1]:.3f}")
    plt.show()
    print(log_fit)
