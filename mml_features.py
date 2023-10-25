import os
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


def generate_sldiwi_file(input_file, save_path):
    input_df = pd.read_csv(input_file)
    result = open(save_path, 'w')
    for i, row in input_df.iterrows():
        result.write("|".join([str(i), str(row['fall_description'])]) + "\n")
    result.close()


def get_semantic_type_str():
    semantic_type_lines = open("./lexicons/mml_semantic_types.txt").readlines()
    semantic_types = [line.split("|")[0].strip() for line in semantic_type_lines]
    return ",".join(semantic_types)


def retrieve_mml_tags(input_path, output_path):
    currDir = os.getcwd()
    mmlrestclient_path = os.path.join(currDir, "mmlrestclient.py")
    semantic_types = get_semantic_type_str()
    cmd = f"python {mmlrestclient_path} https://ii.nlm.nih.gov/metamaplite/rest/annotate {input_path} \
        --output {output_path} --docformat sldiwi --resultformat mmi --semantic-types {semantic_types}"
    cmdOut = os.system(cmd)


def get_cuis_from_mmi_file(file_path):
    cuis_dict = defaultdict(list)
    file_lines = open(file_path).readlines()
    for line in file_lines:
        line_split = line.split("|")
        if len(line_split) > 4:
            cuis_dict[int(line_split[0])].append(line_split[4])
    return cuis_dict


def fit_cui_vectorizer(train_inds, test_inds, cuis_dict, max_features=None):
    cui_strings_train = []
    for ind in train_inds:
        cui_strings_train.append(" ".join(cuis_dict[ind]))
    cui_strings_test = []
    for ind in test_inds:
        cui_strings_test.append(" ".join(cuis_dict[ind]))
    cui_vectorizer = CountVectorizer(max_features=max_features)
    train_vectors = cui_vectorizer.fit_transform(cui_strings_train).toarray()
    test_vectors = cui_vectorizer.transform(cui_strings_test).toarray()
    return train_vectors, test_vectors, cui_vectorizer


def get_mml_cui_CV_features(train_inds, val_inds):
    cuis_dict = get_cuis_from_mmi_file("./ann/train_mml.txt")
    train_vectors, test_vectors, cui_vectorizer = fit_cui_vectorizer(train_inds, val_inds, cuis_dict)
    return train_vectors, test_vectors

def get_mml_cui_train_test_features(n_train_texts, n_test_texts):
    cui_dict_train = get_cuis_from_mmi_file("./ann/train_mml.txt")
    cui_dict_test = get_cuis_from_mmi_file("./ann/test_mml.txt")
    cui_strings_train = []
    for ind in range(n_train_texts):
        cui_strings_train.append(" ".join(cui_dict_train[ind]))
    cui_strings_test = []
    for ind in range(n_test_texts):
        cui_strings_test.append(" ".join(cui_dict_test[ind]))
    cui_vectorizer = CountVectorizer()
    train_vectors = cui_vectorizer.fit_transform(cui_strings_train).toarray()
    test_vectors = cui_vectorizer.transform(cui_strings_test).toarray()
    return train_vectors, test_vectors


if __name__ == '__main__':
    # generate_sldiwi_file("./data/fallreports_2023-9-21_train.csv", "./data/train_sldiwi_for_mml.txt")
    # generate_sldiwi_file("./data/fallreports_2023-9-21_test.csv", "./data/test_sldiwi_for_mml.txt")
    #
    currDir = os.getcwd()
    input_path = os.path.join(currDir, "data", "train_sldiwi_for_mml.txt")
    output_path = os.path.join(currDir, "ann", "train_mml.txt")
    retrieve_mml_tags(input_path, output_path)
    input_path = os.path.join(currDir, "data", "test_sldiwi_for_mml.txt")
    output_path = os.path.join(currDir, "ann", "test_mml.txt")
    retrieve_mml_tags(input_path, output_path)

    # cuis_dict = get_cuis_from_mmi_file('./ann/train_mml.txt')
    # train_vectors, test_vectors, cui_vectorizer = fit_cui_vectorizer(range(300), [], cuis_dict)
    # print(len(cui_vectorizer.vocabulary_.keys()))
    # print(cui_vectorizer.vocabulary_)