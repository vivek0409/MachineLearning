import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib
from apyori import apriori



def main():
    clusterAlgo_Malldata()


def clusterAlgo_Malldata():
    df = pd.read_csv("../../TestData/Mall_Customers.csv")
    print(df)



if __name__== "__main__":
    main()