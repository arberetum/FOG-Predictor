import string

import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine

stemmer = nltk.stem.PorterStemmer()

st = stopwords.words('english')


def gen_verb_strings(texts):
    verb_strings = []
    for t in texts:
        pos_tags = nltk.pos_tag(nltk.word_tokenize(t.lower()))
        verbs = [stemmer.stem(tag[0]) for tag in pos_tags if 'VB' in tag[1]]
        verb_strings.append(" ".join(verbs))
    return verb_strings


def preprocess_texts(texts):
    processed_texts = []
    for t in texts:
        wds = [stemmer.stem(wd) for wd in nltk.word_tokenize(t.lower()) if wd not in st and wd not in string.punctuation]
        processed_texts.append(" ".join(wds))
    return processed_texts


def latent_semantic_analysis(texts, labels):
    texts = np.array(preprocess_texts(texts))
    labels = np.array(labels)
    # generate doc-term matrix
    vectorizer = CountVectorizer()
    vecs = vectorizer.fit_transform(texts).toarray()
    # separate doc-term matrix by class
    lsa_X_class0 = np.transpose(vecs[labels == 0, :])
    lsa_X_class1 = np.transpose(vecs[labels == 1, :])
    # calculate SVDs
    lsa_U_class0, lsa_S_class0, lsa_Vh_class0 = np.linalg.svd(lsa_X_class0, compute_uv=True)
    lsa_U_class1, lsa_S_class1, lsa_Vh_class1 = np.linalg.svd(lsa_X_class1, compute_uv=True)
    # find numbers of dimensions needed to explain 95% of variance
    class0_thresh = 0.95*np.sum(lsa_S_class0**2)
    class0_var_cumsum = np.cumsum(lsa_S_class0**2)
    class0_k = np.argmax(class0_var_cumsum > class0_thresh) + 1  # number of dimensions needed
    class1_thresh = 0.95*np.sum(lsa_S_class1**2)
    class1_var_cumsum = np.cumsum(lsa_S_class1**2)
    class1_k = np.argmax(class1_var_cumsum > class1_thresh) + 1  # number of dimensions needed
    # reduce SVDs
    lsa_Uk_class0 = lsa_U_class0[:, 0:class0_k]
    lsa_Sk_class0 = lsa_S_class0[0:class0_k]
    lsa_Vhk_class0 = lsa_Vh_class0[0:class0_k, :]
    lsa_Uk_class1 = lsa_U_class1[:, 0:class1_k]
    lsa_Sk_class1 = lsa_S_class1[0:class1_k]
    lsa_Vhk_class1 = lsa_Vh_class1[0:class1_k, :]
    return (vectorizer, lsa_X_class0, lsa_X_class1, lsa_Uk_class0, lsa_Sk_class0, lsa_Vhk_class0,
            lsa_Uk_class1, lsa_Sk_class1, lsa_Vhk_class1)


def get_term_importance(term_ind, lsa_X, lsa_Vhk):
    ref_vec = np.matmul(lsa_Vhk, lsa_X[term_ind, :])
    sum_cos_dist = 0
    for i in range(lsa_X.shape[0]):
        # skip the term of interest
        if i == term_ind:
            continue
        this_vec = np.matmul(lsa_Vhk, lsa_X[i, :])
        sum_cos_dist += cosine(ref_vec, this_vec)
    return sum_cos_dist / (lsa_X.shape[0]-1)  # average cosine distance to all other terms in vocab


def gen_verb_importance_vec(full_vectorizer, verb_vectorizer, lsa_X, lsa_Vhk):
    full_vocab = full_vectorizer.vocabulary_
    verb_vocab = verb_vectorizer.vocabulary_
    importance_vector = np.zeros((len(verb_vocab),))
    for stemmed_wd in full_vocab.keys():
        if stemmed_wd in verb_vocab.keys():
            importance_vector[verb_vocab[stemmed_wd]] = get_term_importance(full_vocab[stemmed_wd], lsa_X, lsa_Vhk)
    return importance_vector


def get_prsf_features(train_texts, train_labels, test_texts):
    # get SVDs of doc-term matrices separated by class
    (vectorizer, lsa_X_class0, lsa_X_class1, lsa_Uk_class0, lsa_Sk_class0, lsa_Vhk_class0,
     lsa_Uk_class1, lsa_Sk_class1, lsa_Vhk_class1) = latent_semantic_analysis(train_texts, train_labels)
    # count verb occurrences
    train_verb_strings = gen_verb_strings(train_texts)
    test_verb_strings = gen_verb_strings(test_texts)
    verb_vectorizer = CountVectorizer()
    train_verb_counts = verb_vectorizer.fit_transform(train_verb_strings).toarray()
    test_verb_counts = verb_vectorizer.transform(test_verb_strings).toarray()
    verb_importances_class0 = gen_verb_importance_vec(vectorizer, verb_vectorizer, lsa_X_class0, lsa_Vhk_class0)
    verb_importances_class1 = gen_verb_importance_vec(vectorizer, verb_vectorizer, lsa_X_class1, lsa_Vhk_class1)
    # take elementwise products of verb importances and verb counts
    train_prsf_class0 = np.multiply(train_verb_counts, np.tile(verb_importances_class0, (len(train_texts), 1)))
    train_prsf_class1 = np.multiply(train_verb_counts, np.tile(verb_importances_class1, (len(train_texts), 1)))
    test_prsf_class0 = np.multiply(test_verb_counts, np.tile(verb_importances_class0, (len(test_texts), 1)))
    test_prsf_class1 = np.multiply(test_verb_counts, np.tile(verb_importances_class1, (len(test_texts), 1)))
    train_prsf = np.concatenate((train_prsf_class0, train_prsf_class1), axis=1)
    test_prsf = np.concatenate((test_prsf_class0, test_prsf_class1), axis=1)
    return train_prsf, test_prsf


def __by_importance(item):
    return item[1]


def get_most_important_verbs(train_texts, train_labels, n_most_important):
    # get SVDs of doc-term matrices separated by class
    (vectorizer, lsa_X_class0, lsa_X_class1, lsa_Uk_class0, lsa_Sk_class0, lsa_Vhk_class0,
     lsa_Uk_class1, lsa_Sk_class1, lsa_Vhk_class1) = latent_semantic_analysis(train_texts, train_labels)
    # count verb occurrences
    train_verb_strings = gen_verb_strings(train_texts)
    verb_vectorizer = CountVectorizer()
    train_verb_counts = verb_vectorizer.fit_transform(train_verb_strings).toarray()
    verb_importances_class0 = gen_verb_importance_vec(vectorizer, verb_vectorizer, lsa_X_class0, lsa_Vhk_class0)
    verb_importances_class1 = gen_verb_importance_vec(vectorizer, verb_vectorizer, lsa_X_class1, lsa_Vhk_class1)
    zipped_list_class0 = []
    zipped_list_class1 = []
    for word in verb_vectorizer.vocabulary_.keys():
        zipped_list_class0.append((word, verb_importances_class0[verb_vectorizer.vocabulary_[word]]))
        zipped_list_class1.append((word, verb_importances_class1[verb_vectorizer.vocabulary_[word]]))
    zipped_list_class0 = sorted(zipped_list_class0, key=__by_importance, reverse=True)
    zipped_list_class1 = sorted(zipped_list_class1, key=__by_importance, reverse=True)
    return zipped_list_class0[0:n_most_important], zipped_list_class1[0:n_most_important]



if __name__ == '__main__':
    train_df = pd.read_csv("./data/fallreports_2023-9-21_train.csv")
    train_texts = train_df['fall_description'].to_numpy(dtype=str)
    train_labels = train_df['fog_q_class'].to_numpy(dtype=int)
    (vectorizer, lsa_X_class0, lsa_X_class1, lsa_Uk_class0, lsa_Sk_class0, lsa_Vhk_class0,
                lsa_Uk_class1, lsa_Sk_class1, lsa_Vhk_class1) = latent_semantic_analysis(train_texts, train_labels)
    print(f"Vocab Size: {len(vectorizer.vocabulary_)}")
    print(vectorizer.vocabulary_)
    print(lsa_Sk_class0.shape[0])
    print(lsa_Sk_class1.shape[0])
    print(get_term_importance(18, lsa_X_class0, lsa_Vhk_class0))
    print([word for word in vectorizer.vocabulary_.keys() if vectorizer.vocabulary_[word] == 18])

    # class0_verbs, class1_verbs = get_most_important_verbs(train_texts, train_labels, 25)
    # print("Most Important Class 0 Verbs:")
    # print(class0_verbs)
    # print("Most Important Class 1 Verbs:")
    # print(class1_verbs)

    train_prsf, test_prsf = get_prsf_features(train_texts, train_labels, train_texts)
    print(train_prsf.shape)