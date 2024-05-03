import argparse
import csvq
import pandas as pd

# read data file
def read_data(data_file):
    data = pd.read_csv(data_file, sep=';', header=None, names=['x','y','z'])
    return data

df = read_data("test.csv")

# detect if one point dominates any others=>not dominated by other
def check_dominated(p1, p2, to_max, to_min):
    # dominated score
    dominated = 0

    # check if in factors(higher is better) p1 is dominated by p2, if yes, dominated score plus one
    for f in to_max:
        if p1[f] < p2[f]:
            dominated+=1
    
    # check if in factors(lower is better) p1 is dominated by p2, if yes, dominated score plus one
    for f in to_min:
        if p1[f] > p2[f]:
            dominated+=1

    return dominated


def find_skyline_points(dataset):
    skyline = []

    # compare each point with all other points to check if it is dominated, if not add to skyline
    for index,row in dataset.iterrows():
        dominated_by = False
        for index2,row2 in dataset.iterrows():
            if index != index2:
                # if point is dominated by any point, then break
                if check_dominated(row, row2, ['x','y'], ['z'])==3:
                    dominated_by = True
                    break
        if not dominated_by:
            skyline.append(row)
        
    return pd.DataFrame(skyline)


skyline = find_skyline_points(df)
print(skyline)


parser = argparse.ArgumentParser(description='Skyline Operator')
parser.add_argument('--input', type=str, help='input data')
parser.add_argument('--output', type=str, help='output data')
args = parser.parse_args()
