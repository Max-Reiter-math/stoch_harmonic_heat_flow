import pandas as pd
import numpy as np
import argparse
import json
from rich.console import Console
from rich.table import Table
from sim.utils.table import print_dataframe

"""
Evaluates the mean, median, 10,25,75,90% quantiles over all columns in each row.
"""



def parse_args():
    parser = argparse.ArgumentParser(description="Reads values of a collection of simulations and returns a cross table.")
    parser.add_argument("-path", type=str, help="Path to the directory containing merged simulation data files.")
    parser.add_argument("--copy_to_clipboard", "-ctc", action='store_true', help="Copy the DataFrame to clipboard")
    parser.add_argument("-latex", type=int, help="Print Latex Version of table or not.", default = 1)
    parser.add_argument("-csv", type=int, help="Print Latex Version of table or not.", default = 1)
    parser.add_argument("-savetocsv", type=int, help="Print Latex Version of table or not.", default = 1)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    input_df  = pd.read_csv(args.path, index_col="time")
    
    df = pd.DataFrame({
        "mean": input_df.mean(axis=1),
        "median": input_df.median(axis=1),
        "q10": input_df.quantile(0.10, axis=1),        
        "q25": input_df.quantile(0.25, axis=1),        
        "q75": input_df.quantile(0.75, axis=1),
        "q90": input_df.quantile(0.90, axis=1),
    }, index=input_df.index)

    df.index.name = "time"
            
    if args.copy_to_clipboard:
        df.to_clipboard(index=False)
        print("DataFrame copied to clipboard.")

    if args.latex == 1:
        print(df.to_latex())

    if args.csv == 1:
        print(df.to_csv())

    if args.savetocsv == 1:
        df.to_csv(args.path.replace(".csv", "_stats.csv"), index=True)
        print("DataFrame saved to " + args.path.replace(".csv", "_stats.csv"))

    print_dataframe(df.reset_index())

