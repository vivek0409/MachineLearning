import pandas as pd

def main():
    sample_data = pd.read_csv("../../TestData/Sample_data.csv")

    sample_data['Total1'] = sample_data.loc[:,'Read':'SST'].sum(axis=1)
    sample_data['Total2'] = sample_data.loc[:, 'Read':'SST'].sum(axis=0)

    sample_data.drop('Total2', axis = 1, inplace = True)

    sample_data.insert(11,"Percentage",sample_data.Total1/500*100)
    # print(sample_data)
    # print(sample_data[sample_data["Percentage"]>64])
    print(sample_data.sort_values("Percentage",ascending=False))
    

    # sample_data.to_csv('../TestData/output.csv')

    # print(sample_data.describe())
    # print(sample_data.shape)
    # print(sample_data.info)
    # print(sample_data.loc[:,['Read','SST']])
    # print(sample_data.head())


if __name__== "__main__":
    main()