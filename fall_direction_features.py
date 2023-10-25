import pandas as pd
import numpy as np
from nltk import sent_tokenize, word_tokenize
import re
from collections import defaultdict


def construct_dir_term_dict():
    """Load lexicons of words associated with falling forward, backward, or sideways and return a dictionary with
    the lexicon terms as keys and lists of associated fall directions as values

    :return: dictionary containing terms associated with fall directions as keys and lists of directions ('forward',
    'backward', and/or 'sideways') as values
    """
    forward_terms = open("./lexicons/forward.txt").readlines()
    backward_terms = open("./lexicons/backward.txt").readlines()
    sideways_terms = open("./lexicons/sideways.txt").readlines()
    dir_terms = defaultdict(list)
    for term in forward_terms:
        dir_terms[term.strip().lower()].append('forward')
    for term in backward_terms:
        dir_terms[term.strip().lower()].append('backward')
    for term in sideways_terms:
        dir_terms[term.strip().lower()].append('sideways')
    return dir_terms


def construct_verb_dict():
    """Load a lexicon of verbs that indicate fall direction or are found near indicators of fall direction and return
    a dictionary with the verbs as keys and the associated fall direction as keys (or 'NODIR' for no direction)

    :return: dictionary with verb strings as keys and associate fall directions as values ('NODIR', 'forward',
    'backward', or 'sideways')
    """
    verb_lines = open("./lexicons/verbs.txt")
    verb_dict = dict()
    for line in verb_lines:
        terms = line.split("\t")
        verb_dict[terms[0].strip().lower()] = terms[1].strip()
    return verb_dict


def preprocess_text(text):
    """Lowercase and tokenize the given text

    :param text: string
    :return: list of lowercased sentences in the given text
    """
    lowercased_sentences = [sent.lower() for sent in sent_tokenize(text)]
    return lowercased_sentences


def process_window(match_start, match_end, left_window_size, right_window_size, sentence, term_dict):
    """Look for terms associated with fall direction before and after a key verb in the given sentence

    :param match_start: int, starting index of word to search around in the sentence
    :param match_end: int, ending index of word to search around in the sentence
    :param left_window_size: int, number of words to check before the central/pivot word
    :param right_window_size: int, number of words to check after the central/pivot word
    :param sentence: string, sentence to search in
    :param term_dict: dictionary with terms to look for as keys and associated fall directions as keys
    :return: dictionary with the keys 'forward', 'backward', and 'sideways' and the associated numbers of terms found in
    the window as values
    """
    before_match_wds = word_tokenize(sentence[:match_start])
    after_match_wds = word_tokenize(sentence[match_end:])
    # construct strings of designated window length before and after verb match
    if left_window_size > 0:
        before_string = " ".join(before_match_wds[-1*(min(len(before_match_wds), left_window_size)):])
    else:
        before_string = ""
    after_string = " ".join(after_match_wds[:min(len(after_match_wds), right_window_size)])
    votes = dict()
    votes['forward'] = 0
    votes['backward'] = 0
    votes['sideways'] = 0
    dir_term_regex = re.compile("|".join(term_dict.keys()))
    for match in dir_term_regex.finditer(before_string):
        for direction in term_dict[match.group()]:
            votes[direction] += 1
    for match in dir_term_regex.finditer(after_string):
        for direction in term_dict[match.group()]:
            votes[direction] += 1
    return votes


def find_fall_direction(text, left_window_size, right_window_size):
    """Classify the fall direction described in the given text by counting numbers of direction-associated terms before
    and after key verbs (such as 'fell', 'landed', etc.) and choosing the direction(s) that have the most associated
    terms in the text chunk

    :param text: string, chunk of text to classify
    :param left_window_size: int, number of words to check before key verbs
    :param right_window_size: int, number of words to check after key verbs
    :return: dictionary with the keys 'forward', 'backward', and 'sideways' and 1 if the fall was determined to be in
    each direction and 0 otherwise (it's possible for multiple directions to have 1's when numbers of associated terms
    are tied, and it's possible for all to be 0 when no direction-associated terms near key verbs are identified)
    """
    votes = dict()
    votes['forward'] = 0
    votes['backward'] = 0
    votes['sideways'] = 0
    sentences = preprocess_text(text)
    dir_terms = construct_dir_term_dict()
    verb_dict = construct_verb_dict()
    verb_regex = re.compile('|'.join(verb_dict.keys()))
    for sent in sentences:
        for match in verb_regex.finditer(sent):
            # add a vote if a directional verb was found
            if verb_dict[match.group()] != 'NODIR':
                votes[verb_dict[match.group()]] += 1
            # search window around verb
            window_votes = process_window(match.start(), match.end(), left_window_size, right_window_size, sent,
                                          dir_terms)
            # add new votes
            for key in votes.keys():
                votes[key] += window_votes[key]
    max_vote_count = max(votes.values())
    classes = dict()
    classes['forward'] = 0
    classes['backward'] = 0
    classes['sideways'] = 0
    # choose no class if no key terms were found
    if max_vote_count == 0:
        return classes
    # return list of all classes with the maximum nonzero vote-count
    for key in votes.keys():
        if votes[key] == max_vote_count:
            classes[key] = 1
    return classes


def direction_metrics(truth_df, pred_df):
    fp = 0
    tp = 0
    fn = 0
    for row_num in range(len(truth_df.index)):
        for col_num in range(len(truth_df.columns)):
            if ((truth_df.iloc[row_num, col_num] == pred_df.iloc[row_num, col_num]) and
                    (truth_df.iloc[row_num, col_num] == 1)):
                tp += 1
            elif ((truth_df.iloc[row_num, col_num] != pred_df.iloc[row_num, col_num]) and
                  (truth_df.iloc[row_num, col_num] == 1)):
                fn += 1
            elif ((truth_df.iloc[row_num, col_num] != pred_df.iloc[row_num, col_num]) and
                  (truth_df.iloc[row_num, col_num] == 0)):
                fp += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1


def get_all_fall_directions(texts, left_window_size=5, right_window_size=7):
    """Classify the fall directions described in the given texts by counting numbers of direction-associated terms before
    and after key verbs (such as 'fell', 'landed', etc.) and choosing the direction(s) that have the most associated
    terms in each text chunk

    :param texts: list of strings, chunks of text to individually classify
    :param left_window_size: int, number of words to check before key verbs
    :param right_window_size: int, number of words to check after key verbs
    :return: dataframe with the columns 'forward', 'backward', and 'sideways'; indices corresponding to the indices
    in the given texts array; and class labels as 0 and 1 accordingly (see find_fall_direction)
    """
    fall_directions_df = pd.DataFrame(columns=['forward', 'backward', 'sideways'])
    for text in texts:
        fall_directions_df = pd.concat( [fall_directions_df, pd.DataFrame(
            find_fall_direction(text, left_window_size, right_window_size), index=[0])] )
    return fall_directions_df


def load_fall_direction_truth():
    truth_dir_df = pd.read_csv("./ann/fallreports_2023-9-21_train_dir.csv")
    truth_dir_df.drop(columns=['record_id', 'fall_description'], inplace=True)
    truth_dir_df.replace('', 0, inplace=True)
    return truth_dir_df


def grid_search_window_size():
    """Perform a grid search over forward and backward window sizes (numbers of words to check before and after key
    verbs) to determine best window sizes for determining fall direction
    """
    left_window_range = range(11)
    right_window_range = range(11)
    f1s = np.zeros((len(left_window_range), len(right_window_range)))
    truth_df = load_fall_direction_truth()
    texts_df = pd.read_csv("./ann/fallreports_2023-9-21_train_dir.csv")
    texts_array = texts_df['fall_description'].to_numpy(dtype=str)
    for i, left_window_size in enumerate(left_window_range):
        print(f"Searching left window size {(i+1)}/{len(left_window_range)}")
        for j, right_window_size in enumerate(right_window_range):
            pred_df = get_all_fall_directions(texts_array, left_window_size, right_window_size)
            precision, recall, f1 = direction_metrics(truth_df, pred_df)
            f1s[i, j] = f1
    best_ind = np.unravel_index(np.argmax(f1s), f1s.shape)
    print(f"Best left window length: {left_window_range[best_ind[0]]}")
    print(f"Best right window length: {right_window_range[best_ind[1]]}")
    print(f"Best f1: {f1s[best_ind]}")


if __name__ == '__main__':
    # dir_terms = construct_dir_term_dict()
    # print(dir_terms)
    # verb_dict = construct_verb_dict()
    # print(verb_dict)
    # example_text = "The patient was home playing catch in backyard. He lost balance after jumping up to catch a ball that was thrown overhead and fell backwards on his backside. He rolled over and pushed himself up"
    # classes = find_fall_direction(example_text, left_window_size=3, right_window_size=6)
    # print(classes)
    # grid_search_window_size()

    # get distribution of falls from manual annotations
    train_data = pd.read_csv("./data/fallreports_2023-9-21_train.csv")
    train_data = train_data['fog_q_class']
    ann_data = pd.read_csv("./ann/fallreports_2023-9-21_train_dir.csv")
    ann_data.drop(columns=['record_id', 'fall_description'], inplace=True)
    ann_data.replace('', 0, inplace=True)
    train_data = pd.concat([train_data, ann_data], axis=1)
    print(train_data.groupby(['fog_q_class']).sum())
