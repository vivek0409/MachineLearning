import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn import tree

# creditdata = pd.read_csv("../TestData/Credit_Data.csv")
def main():
    decisionTree_SocialNetworkData()


def decisionTree_SocialNetworkData():
    ssd = pd.read_csv("../../TestData/decision_tree_classification_social_Network_ads.csv")

    df = pd.DataFrame(ssd)
    print(df)

    X = df.iloc[:,:2].values
    y= df.iloc[:,2:].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    # print(y_predict)

    accuracy = accuracy_score(y_predict, y_test)
    print(accuracy)

    # new_cust = [[46, 41000]]
    # print(classifier.predict(new_cust))

    # tree.plot_tree(classifier)
    # plt.show()


def decisionTreeSampleData():
    data = {
        'Income': [50000, 60000, 75000, 40000, 80000, 55000, 70000, 45000],
        'Debt': [20000, 15000, 25000, 10000, 30000, 22000, 18000, 12000],
        'CreditHistory': [1, 2, 1, 3, 2, 1, 2, 3],  # 1: excellent, 2: good, 3: fair
        'Risk': ['Low', 'Low', 'Medium', 'High', 'Medium', 'Low', 'Medium', 'High']
    }
    df = pd.DataFrame(data)
    # print(df)
    # print(df.dtypes)
    # print(df.describe())
    # print(list(df.columns.values))
    # print(df.isnull().sum())
    X = df.iloc[:, :3]
    # print(X)
    y = df.iloc[:, 3:]
    # print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # print(X_train)
    # print(X_test)
    #
    # print(y_train)
    # print(y_test)

    classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    # print(y_predict)

    accuracy = accuracy_score(y_predict, y_test)
    print(accuracy)

    ner_cust = [[50000, 16000, 2]]
    print(classifier.predict(ner_cust))

    tree.plot_tree(classifier)
    plt.show()


if __name__== "__main__":
    main()