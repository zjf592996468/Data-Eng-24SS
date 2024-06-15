import ast
import re
import time
from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# download nltk stopwords
nltk.download('stopwords')

# load dataset
data = pd.read_csv('cora.tsv', sep='\t')

# union nltk and sklearn stopwords
stop_words = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))


# remove all special characters
def remove_special_characters(text):
    return re.sub(r'[^A-Za-z0-9\s]', '', text.lower())


# preprocess dataset
def preprocess_data(df):
    df.fillna('', inplace=True)  # fill NAN with blank
    df = df.astype(str).map(remove_special_characters)  # remove special characters

    return df


data = preprocess_data(data)


# tokenize text
def tokenize(text):
    # uniform all letters to lowercase and split them into a set
    tokens = text.split()
    tokens = set(token for token in tokens if token not in stop_words)

    return tokens


# create blocks by attributes in dataset
def create_blocks(data, attributes):
    blocks = defaultdict(set)
    for index, row in data.iterrows():
        for attr in attributes:
            text = row[attr]
            tokens = tokenize(text)
            for token in tokens:
                blocks[token].add(row['id'])

    return blocks


# filter blocks by the size of the block
def filter_blocks(blocks, min_size=2, max_size=100):
    filtered_blocks = {token: ids for token, ids in blocks.items() if min_size <= len(ids) <= max_size}

    return filtered_blocks


# create inverted index without 'id'
attributes_list = data.columns.drop('id').tolist()

# create and filter blocks
blocks = create_blocks(data, attributes_list)
blocks = filter_blocks(blocks)


# get all probable pairs
def prob_pairs(blocks):
    pairs = set()
    for token, ids in blocks.items():
        ids_list = list(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if ids_list[i] != ids_list[j]:
                    pairs.add((ids_list[i], ids_list[j]))

    return pairs


pairs = prob_pairs(blocks)


# get values of attributes
def get_values(data, id1, id2, attr_list):
    pair_value1 = ' '.join(data.loc[data['id'] == id1, attr_list].values[0])
    pair_value2 = ' '.join(data.loc[data['id'] == id2, attr_list].values[0])

    return pair_value1, pair_value2


# calculate jaccard similarity
def jaccard_sim(str1, str2):
    set1 = tokenize(str1)
    set2 = tokenize(str2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union != 0 else 0


# check duplicate
def check_duplicate(data, pairs, threshold=0.5):
    duplicates = set()
    print("start checking duplicates...")
    for pair in pairs:
        author = get_values(data, pair[0], pair[1], ['authors'])
        title = get_values(data, pair[0], pair[1], ['booktitle', 'journal', 'title'])
        date = get_values(data, pair[0], pair[1], ['date', 'month', 'year'])
        sim = np.array(
            [jaccard_sim(author[0], author[1]), jaccard_sim(title[0], title[1]), jaccard_sim(date[0], date[1])])
        if np.mean(sim) > threshold:
            duplicates.add(pair)

    return duplicates


# save and load results
def save_detected_pairs(pairs, filename='duplicates.txt'):
    with open(filename, 'w') as file:
        for pair in pairs:
            file.write(f"{pair}\n")


def load_detected_pairs(filename='duplicates.txt'):
    detected_pairs = set()
    with open(filename, 'r') as file:
        for line in file:
            detected_pairs.add(tuple(map(int, ast.literal_eval(line.strip()))))

    return detected_pairs


# load true duplicates
true_pairs_df = pd.read_csv('cora_DPL.tsv', sep='\t')
true_pairs = set((row['id1'], row['id2']) for _, row in true_pairs_df.iterrows())


# optional: evaluate result
def evaluate(true_pairs, predicted_pairs):
    tp = len(true_pairs & predicted_pairs)
    fp = len(predicted_pairs - true_pairs)
    fn = len(true_pairs - predicted_pairs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure


# check duplicates
t_start = time.perf_counter()
duplicate_pairs = check_duplicate(data, pairs)
t_end = time.perf_counter()

# record runtime
runtime = t_end - t_start
print(f"Runtime: {runtime:0.4f} seconds")

# save results as txt
save_detected_pairs(duplicate_pairs)

# evaluate results
detected_pairs = load_detected_pairs()
precision, recall, f_measure = evaluate(true_pairs, detected_pairs)
print(f"Precision: {precision:0.4f}, Recall: {recall:0.4f}, F-measure: {f_measure:0.4f}")
