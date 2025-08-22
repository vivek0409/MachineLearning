import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix

dataset = pd.read_csv("../../TestData/iris.csv")

def main():
    seaboarnPackage()


def seaboarnPackage():
    # sb.get_dataset_names()
    df = sb.load_dataset('diamonds')
    # print(dataset)
    # sb.histplot(df['price'],kde = True)
    sb.pairplot(dataset, hue='Species')
    plt.show()

def corelationpart2():
    df = pd.DataFrame({'var_1': np.random.randint(0, 100, 500)})
    df['var_2'] = df['var_1'] + np.random.normal(0, 20, 500)  ## positively correlated with var_1
    df['var_3'] = 100 - df['var_1'] + np.random.normal(0, 20, 500)  ##negatively correlated with var_1
    print(df)
    print(df.corr())

    scatter_matrix(df)
    plt.show()

def coRelation():
    x  = np.random.randint(0,100,500)
    y  = 100-x + np.random.normal(1,10,500)
    z = np.random.randint(0, 100, 500)
    print(np.corrcoef(x,z))
    plt.scatter(x,z)
    plt.show()

def scatterPlot():
    x = [1, 2, 3, 4, 5]
    y = [5, 7, 4, 6, 8]
    dataset = [x,y]
    # sns.scatterplot(dataset)
    plt.scatter(x,y)
    plt.grid()
    plt.show()

def boxplotimplementation():
    d1 = np.random.normal(100,50,1000)
    d2 =  np.random.normal(-4,500,1000)
    # print(data)
    dataset = [d1,d2]

    plt.boxplot(dataset)

    plt.show()

def histogram():
    randon_int = np.random.randint(1,100, size = 100)
    print(randon_int)

    plt.hist(randon_int, bins = 10)
    plt.show()

def visualizeData():
    print(dataset)

    # sb.scatterplot(x='Sepal.Length' , y ='Sepal.Width' , hue = 'Species', data = dataset)
    # plt.show()
    # sb.countplot(y='Species',data = dataset)
    # plt.show()
    # subjects = ['maths', 'computers', 'english', 'Accounts', 'Economics']
    # counts = [100, 40, 35, 29, 20]
    #
    # plt.title('Subject Count')
    # plt.xlabel('Counts')
    # plt.ylabel('Subjects')
    # plt.barh(subjects,counts)
    # plt.show()

    data = [27, 45, 36, 18, 72, 81, 108, 19, 9]
    x = range(len(data))

    plt.title('Count')
    plt.xlabel('Index')
    plt.ylabel('Count')
    plt.bar(x, data)
    plt.show()

def barchartImplementation():
    fig, ax = plt.subplots()

    fruits = ['apple', 'blueberry', 'cherry', 'orange']
    counts = [40, 100, 30, 55]
    bar_labels = ['red', 'blue', '_red', 'orange']
    bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

    ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

    ax.set_ylabel('fruit supply')
    ax.set_title('Fruit supply by kind and color')
    ax.legend(title='Fruit color')

    plt.show()

def stackBarChart():
    students = ['Minal', 'Mohit', 'Mukul', 'Pradeep']
    Test1 = np.array([18, 14, 12, 15])
    Test2 = np.array([20, 18, 16, 19])
    Test3 = np.array([16, 19, 17, 18])
    index = students
    plt.bar(index, Test1, width=0.5, label='Test1', color='green', bottom=Test2 + Test3)
    plt.bar(index, Test2, width=0.5, label='Test2', color='red', bottom=Test3)
    plt.bar(index, Test3, width=0.5, label='Test3', color='blue')
    plt.ylabel("Marks")
    plt.xlabel("Students")
    plt.legend(loc='upper right')
    plt.title("Marks obtained in three subjects")
    plt.show()

if __name__== "__main__":
    main()