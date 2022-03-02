import pandas as pd
import numpy as np 
import argparse


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Tokenize given raw labels for RoBERTa and saves it.')
    # Input args
    parser.add_argument('--input_data', action='store', help='Path from where dataframe to be processed should be loaded.')
    # Output args
    parser.add_argument('--out_data', action='store', help='Path where tokenized data should be saved.')
    
    args = parser.parse_args()
    
    import pandas as pd; pd.read_csv(args.input_data, index_col=0)["Y"].to_csv(args.out_data, header=None, index=None)