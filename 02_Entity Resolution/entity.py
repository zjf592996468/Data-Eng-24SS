import time

import pandas as pd

# load dataset
data = pd.read_csv('cora.tsv', sep='\t')


# preprocess dataset
def preprocess_data(df):
    df.fillna('', inplace=True)  # fill NAN with blank
    return df


data = preprocess_data(data)


# # calculate cosine similarity
# def cos_sim(df):
#     # todo: calculate cos similarity
#
#     return cos_sim
#
#
# sim_matrix = cos_sim(data)


# calculate jaccard similarity
def jaccard_sim(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


# check duplicate
def check_duplicate(df, threshold=0.5):
    duplicates = []
    print("start checking duplicates...")
    # todo: detect all duplicates according to sim_matrix
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            str1 = df.iloc[i]['authors'] + ' ' + df.iloc[i]['title']  # 感觉这里还不太够，只判断作者和题目
            str2 = df.iloc[j]['authors'] + ' ' + df.iloc[j]['title']
            sim = jaccard_sim(str1, str2)
            if sim > threshold:
                duplicates.append((i + 1, j + 1))
    return duplicates


t_start = time.perf_counter()
duplicate_pairs = check_duplicate(data)
t_end = time.perf_counter()

# save as txt
with open('duplicates.txt', 'w') as file:
    for pair in duplicate_pairs:
        file.write(f"{pair}\n")
    file.close()

# record runtime
runtime = t_end - t_start
print(f"Runtime: {runtime:0.4f} seconds")
