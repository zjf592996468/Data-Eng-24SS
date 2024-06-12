import time

import pandas as pd

# load dataset
data = pd.read_csv('cora.tsv', sep='\t')


# preprocess dataset
def preprocess_data(df):
    df.fillna('', inplace=True)  # fill NAN with blank
    return df


data = preprocess_data(data)


# calculate similarity
def cal_sim(df):
    # todo: calculate similarity - cos, jaccard
    return sim


sim_matrix = cal_sim(data)


# check duplicate
def check_duplicate(df):
    duplicates = []
    # todo: detect all duplicates according to sim_matrix
    return duplicates


t_start = time.perf_counter()
duplicate_pairs = check_duplicate(sim_matrix)
t_end = time.perf_counter()

# save as txt

# record runtime
runtime = t_end - t_start
print(f"Runtime: {runtime:0.4f} seconds")
