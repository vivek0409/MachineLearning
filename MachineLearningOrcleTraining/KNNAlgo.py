from sklearn.datasets import load_iris
from sklearn import preprocessing
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score


def main():
    readingViaCSVFile()

def readingViaCSVFile():
    dataset = pd.read_csv("../TestData/iris.csv")
    # print(dataset.shape)
    # print(dataset.describe())
    # print(dataset.info())
    # print(dataset.groupby('Species').size())
    feature_names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
    X = dataset[feature_names].values
    # print(X)

    # Normalizing or scaling the data
    normalize_X = sk.preprocessing.normalize(X)
    # print(normalize_X)

    y = dataset['Species'].values

    # ENcoding Y so that it will have numeric values
    le = LabelEncoder()
    y = le.fit_transform(y)
    # print(y)

    # Training Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # print(X_train.size)

    # KNN Model
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    my_y_pred = knn.predict(X_test)
    print(my_y_pred)

    # Calculating Accuracy
    accuracy = accuracy_score(y_test, my_y_pred) * 100
    print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

def withoutUsingCSVFile():
    iris = load_iris()

    # Create DataFrame similar to reading from CSV
    feature_names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
    df = pd.DataFrame(iris.data, columns=feature_names)
    df['Species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Features and target
    X = df[feature_names].values
    normalize_X = preprocessing.normalize(X)
    y = df['Species'].values

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # KNN Model
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    my_y_pred = knn.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, my_y_pred) * 100

    print("Predicted output for test data:", my_y_pred)
    print(f"Accuracy of our model is equal {round(accuracy, 2)} %.")



if __name__== "__main__":
    main()