import argparse
import pathlib

import pandas as pd

# parse args from terminal
parser = argparse.ArgumentParser(description='Skyline Operator')
parser.add_argument('--input', type=str, help='Absolute path of input data file.')
parser.add_argument('--output', type=str, help='Absolute path of output data file.')
args = parser.parse_args()

# path of data
path_in = pathlib.Path(args.input)
path_out = pathlib.Path(args.output)


# read data file
def read_data(data_file):
    data = pd.read_csv(data_file, sep=';', header=None, names=['x', 'y', 'z'])
    return data


df = read_data(path_in)


# check if p1 is dominated by p2
def check_dominated(p1, p2, to_max, to_min):
    # set a dominated score
    dominated_score = 0

    # check if in factors(higher is better) p1 is dominated by p2, if yes, dominated score plus one
    for f in to_max:
        if p1[f] <= p2[f]:
            dominated_score += 1

    # check if in factors(lower is better) p1 is dominated by p2, if yes, dominated score plus one
    for f in to_min:
        if p1[f] >= p2[f]:
            dominated_score += 1

    # if dominated score is 3, then p1 is dominated by p2
    if dominated_score == 3:
        return True
    else:
        return False


def find_skyline_points(dataset):
    skyline = []

    # compare each point with all other points to check if it is dominated, if not add to skyline
    for index, row in dataset.iterrows():
        dominated_by = False
        for index2, row2 in dataset.iterrows():
            if index != index2:
                # if point is dominated by any point, then break
                if check_dominated(row, row2, ['x', 'y'], ['z']):
                    dominated_by = True
                    break
        if not dominated_by:
            skyline.append(row)

    return pd.DataFrame(skyline)


skyline = find_skyline_points(df)


def write_data(dataframe, path):
    dataframe.to_csv(path, index=False, header=False, sep=';')


write_data(skyline, path_out)
