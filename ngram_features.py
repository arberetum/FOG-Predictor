import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


st = stopwords.words('english')
stemmer = PorterStemmer()


def preprocess_texts(texts):
    processed_texts = []
    for text in texts:
        these_stemmed_tokens = [stemmer.stem(wd) for wd in word_tokenize(text.lower()) if wd not in st and
                        wd not in string.punctuation]
        processed_texts.append(" ".join(these_stemmed_tokens))
    return processed_texts


def get_ngrams_features(train_texts, test_texts):
    train_texts_pre = preprocess_texts(train_texts)
    test_texts_pre = preprocess_texts(test_texts)
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    train_feat = vectorizer.fit_transform(train_texts_pre).toarray()
    test_feat = vectorizer.transform(test_texts_pre).toarray()
    return train_feat, test_feat