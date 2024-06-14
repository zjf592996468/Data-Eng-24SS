import pandas as pd
import xml.etree.ElementTree as ET
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
# Get the list of stop words
stop_words = set(stopwords.words('english'))

# load the data from the file
with open('cora.tsv') as f:
    data = pd.read_csv(f, sep='\t')

data = data.astype(str)

# tokenize the data
def tokenize(text):
    text = text.strip(" `,.()")
    text = text.lower()
    tokens = text.split()
    # get rid of stopwords
    tokens = [token.strip(",") for token in tokens if token not in stop_words]
    return set(tokens)

# create blocks, tag_name is the name of the tag that contains the text content, here is ""
def build_blocks(data, attributes:list):
    blocks = defaultdict(set)
    for index,row in data.iterrows():
        for attribute in attributes:
            text_content = row[attribute]
            tokens = tokenize(text_content)
            # print(tokens)
            for token in list(tokens):
                blocks[token].add(row['id'])
    return blocks

# filter the blocks by the size of the block
def filter_blocks(blocks, min_size=2, max_size=100):
    filtered_blocks = {token: ids for token, ids in blocks.items() if min_size <= len(ids) <= max_size}
    return filtered_blocks

# create inverted index
attribute_list = [att for att in data.columns.tolist() if att != 'id']
# biuld blocks
blocks = build_blocks(data, attribute_list)
blocks = filter_blocks(blocks)

def get_comparison_pairs(blocks):
    pairs = []
    for token, elements in blocks.items():
        for i in range(len(elements)):
            for j in range(i+1,len(elements)):
                elements_list = list(elements)
                if elements_list[i] != elements_list[j]:
                    pairs.append((elements_list[i], elements_list[j]))
    return pairs

def remove_redundant_pairs(pairs):
    unique_pairs = set()
    for pair in pairs:
        if pair[::-1] not in unique_pairs:
            unique_pairs.add(pair)
    return unique_pairs

pairs = get_comparison_pairs(blocks)
pairs = remove_redundant_pairs(pairs)

def get_attributes_values(data, id_A, id_B, attribute_list):
    pair_a_value = ' '.join(data.loc[data['id'] == id_A, attribute_list].values[0])
    pair_b_value = ' '.join(data.loc[data['id'] == id_B, attribute_list].values[0])
    return (pair_a_value, pair_b_value)

def jacard_similarity(text1, text2):
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

dupulicate_pairs = []

def find_duplicates(data, pairs):
    for pair in pairs:
        author = get_attributes_values(data, pair[0], pair[1], ['authors'])
        title = get_attributes_values(data, pair[0], pair[1], ['booktitle', 'journal', 'title'])
        date = get_attributes_values(data, pair[0], pair[1], ['date','month','year'])
        author_similarity_score = jacard_similarity(author[0], author[1])
        title_similarity_score = jacard_similarity(title[0], title[1])
        date_similarity_score = jacard_similarity(date[0], date[1])
        total_similarity_score = (author_similarity_score + title_similarity_score + date_similarity_score) / 3
        if total_similarity_score > 0.6:
            dupulicate_pairs.append(pair)
    return dupulicate_pairs

# 19111 duplicates in result
dupulicate_pairs = find_duplicates(data, pairs)