from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag_sents
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer

def get_verb_synset_strings(texts):
    synset_strings = []
    for text in texts:
        sentences = sent_tokenize(text)
        tokenized_sentences = []
        for sentence in sentences:
            tokenized_sentences.append(word_tokenize(sentence))
        pos_tags = pos_tag_sents(tokenized_sentences)
        this_synset_string = ""
        for sent_num in range(len(pos_tags)):
            for tagged_word in pos_tags[sent_num]:
                if 'VB' in tagged_word[1]:
                    this_term_syns = [synset._name for synset in wn.synsets(tagged_word[0], pos=wn.VERB)]
                    this_synset_string += " ".join(this_term_syns) + " "
        this_synset_string = this_synset_string.strip()
        synset_strings.append(this_synset_string)
    return synset_strings


def get_verb_synset_features(train_texts, test_texts):
    train_synset_strings = get_verb_synset_strings(train_texts)
    test_synset_strings = get_verb_synset_strings(test_texts)
    vectorizer = CountVectorizer(tokenizer=__synset_splitter, token_pattern=None)
    train_features = vectorizer.fit_transform(train_synset_strings).toarray()
    test_features = vectorizer.transform(test_synset_strings).toarray()
    return train_features, test_features, vectorizer


def __synset_splitter(text):
    return text.split()


if __name__ == '__main__':
    train_df = pd.read_csv("./data/fallreports_2023-9-21_train.csv")
    train_texts = train_df['fall_description'].to_numpy(dtype=str)
    train_labels = train_df['fog_q_class'].to_numpy(dtype=int)
    print(get_verb_synset_strings(train_texts))
    train_feat, train_feat2, vectorizer = get_verb_synset_features(train_texts, train_texts)
    print(train_feat.shape)
    # print(vectorizer.vocabulary_)

