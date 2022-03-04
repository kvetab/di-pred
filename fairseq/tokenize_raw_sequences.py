import pandas as pd
import numpy as np 
from os import path
import argparse


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Tokenize given raw data for RoBERTa and saves it.')
    # Input args
    parser.add_argument('--input_data', action='store', help='Path from where dataframe to be processed should be loaded.')
    parser.add_argument('--input_col_index', action='store', type=int, help='Index of a column from which sequences should be used.')
    # Output args
    parser.add_argument('--out_data', action='store', help='Path where tokenized data should be saved.')
    
    args = parser.parse_args()

    # Load data
    print('Loading data...')
    data = pd.read_csv(args.input_data, index_col=0)

    print('Preprocessing sequences...')
    split_seq_data = data.iloc[:,args.input_col_index].apply(lambda seq: ' '.join(list(seq)))

    print('Saving data...')
    np.savetxt(args.out_data, split_seq_data.values, fmt='%s')