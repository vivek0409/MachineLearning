import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib# Or 'Qt5Agg' if you have PyQt5 installed
from apyori import apriori

def main():
    apriori_samplrDatset()

def apriori_samplrDatset():
    df = pd.read_csv("../../TestData/dataset.csv")
    # print(df)
    # print(df.itemDescription.value_counts())
    # print(df.isnull().sum())
    df.itemDescription.value_counts().head(10).plot.bar()
    plt.title('Top 10 selling products')
    # plt.show()








def aprioriAlso_MarketBasketDataSet():
    df = pd.read_csv("../../TestData/Market_Basket_Optimisation.csv", header= None)
    # print(df)

    df.fillna(0,inplace=True)
    # print(df.shape)

    transaction = df.iloc[:7501, :20].astype(str).values.tolist()

    # for i in range(0, 7500):
    #     transaction.append([str(df.values[i, j]) for j in range(0, 20)])

    # print(transaction)

    rules = apriori(transactions = transaction,min_support = 0.003,min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

    result = pd.DataFrame(list(rules))

    # print(result)

    first_values = []
    second_values = []
    third_values = []
    fourth_values = []

    for i in range(result.shape[0]):
        single_list = result['ordered_statistics'][i][0]
        first_values.append(list(single_list[0]))
        second_values.append(list(single_list[1]))
        third_values.append(single_list[2])
        fourth_values.append(single_list[3])

    # print((first_values))

    lhs = pd.DataFrame(first_values, columns=['lhs'])
    rhs = pd.DataFrame(second_values, columns=['rhs'])
    confidence = pd.DataFrame(third_values, columns=['Confidence'])
    lift = pd.DataFrame(fourth_values, columns=['lift'])

    finalResults = pd.concat([lhs,rhs,confidence,lift],axis = 1)

    print(finalResults)

if __name__== "__main__":
    main()