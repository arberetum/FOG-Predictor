from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from ngram_features import get_ngrams_features
from fall_direction_features import get_all_fall_directions
from mml_features import get_mml_cui_train_test_features
from prsf_features import get_prsf_features
from wordnet_features import get_verb_synset_features


# load data
train_df = pd.read_csv("./data/fallreports_2023-9-21_train.csv")
train_texts = train_df['fall_description'].to_numpy(dtype=str)
train_labels = train_df['fog_q_class'].to_numpy(dtype=int)
test_df = pd.read_csv("./data/fallreports_2023-9-21_test.csv")
test_texts = test_df['fall_description'].to_numpy(dtype=str)
test_labels = test_df['fog_q_class'].to_numpy(dtype=int)

# get feature sets
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

# train model
nb = GaussianNB().fit(train_feat, train_labels)

# evaluate model
test_pred = nb.predict(test_feat)
acc_test = nb.score(test_feat, test_labels)
micro_f1_test = f1_score(test_labels, test_pred, average='micro')
macro_f1_test = f1_score(test_labels, test_pred, average='macro')
print(f"Test Accuracy: {acc_test}")
print(f"Test Micro F1: {micro_f1_test}")
print(f"Test Macro F1: {macro_f1_test}")
# prints the following
# Train Feature Matrix Shape: (299, 13578)
# Test Feature Matrix Shape: (71, 13578)
# Test Accuracy: 0.7605633802816901
# Test Micro F1: 0.7605633802816902
# Test Macro F1: 0.7557174660999797