import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats

dataset = [11, 10, 12, 14, 12, 15, 14, 13, 15, 102, 12, 14, 17, 19, 107, 10, 13, 12, 14, 12, 108, 12, 11, 14, 13,
           15, 10, 15, 12, 10, 14, 13, 15, 10]
def main():
    checkPercentile()

def checkPercentile():
    q1,q3 = np.percentile(dataset,[25,75])
    print(q3)
    print(q1)


def detect_outliers(data):
    outliers =[]
    threshold=3     ## 3 std deviation
    mean=np.mean(data)
    std=np.std(data)
    for i in data:
        z_score=(i-mean)/std
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers

def findOutliers():
    print(detect_outliers(dataset))
    # plt.hist(dataset)
    # plt.show()


def basicStats():
    df = sns.load_dataset('tips')
    print(df['total_bill'])
    # print(stats.mean(df['total_bill']))
    # print(stats.median(df['total_bill']))
    # print(stats.mode(df['total_bill']))
    sns.boxplot(df['total_bill'])
    plt.show()

if __name__== "__main__":
    main()