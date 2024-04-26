import argparse
import csv

parser = argparse.ArgumentParser(description='Skyline Operator')
parser.add_argument('--input', type=str, help='input data')
parser.add_argument('--output', type=str, help='output data')
args = parser.parse_args()
