import pandas as pd
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Layer,Lambda
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    ANNAlgo_concreteDataset()


def ANNAlgo_concreteDataset():

    concrete = pd.read_csv("../../TestData/concrete.csv")
    sc = MinMaxScaler()
    concrete = pd.DataFrame(sc.fit_transform(concrete))
    # print(concrete.describe())

    X = concrete.iloc[:, :-1].values
    # print(X)
    y = concrete.iloc[:,-1:].values
    # print(y)
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=40, activation='relu'))
    model.add(tf.keras.layers.Dense(units=50, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error',)
    model.fit(X_train, y_train, batch_size=50, epochs=100)

    y_pred = model.predict(X_test)

    y_test = y_test.reshape(-1)
    y_pred = y_pred.reshape(-1)

    # Regression scores
    r2 = r2_score(y_test, y_pred)
    print(f"Accuracy Score: {r2:.4f}")

    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test, label='Actual (y_test)', color='blue')
    # plt.plot(y_pred, label='Predicted (y_pred)', color='red')
    # plt.title('Actual vs Predicted Values')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Target Value')
    # plt.legend()
    # plt.show()

    plt.figure(figsize=(12, 8))
    with plt.style.context('fivethirtyeight'):
        plt.plot(sorted(y_test), label='Actual')
        plt.plot(sorted(y_pred), label='Predicted')
        #     plt.fill_between(x=np.arange(0,len(y_pred)),
        #                      y1=np.array(sorted(y_pred)+MAE),
        #                      y2=sorted(y_pred)-MAE,
        #                      alpha=0.1, color='r', label='MAE')

        plt.title('Testing prediction')
        plt.ylabel('Concrete Strength')
        plt.xlabel('Item')
        plt.legend()
    plt.show()


def ANNAlgo_ChurnDataSet():
    anndata = pd.read_csv("../../TestData/Churn_Modelling.csv")
    # print(anndata.shape)
    # print(anndata.describe())
    # print(anndata.isnull().sum())

    X = anndata.iloc[:,3:-1].values
    X[:,2] = LabelEncoder().fit_transform(X[:,2])
    X[:, 1] = LabelEncoder().fit_transform(X[:, 1])
    # print(X)
    y = anndata.iloc[:,-1:].values
    # print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # print(X_train)
    # print(X_test)

    #### Initiation ANN
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add((tf.keras.layers.Dense(units = 6 , activation= 'relu')))
    ann.add(tf.keras.layers.Dense(units=1 , activation='sigmoid'))
    ann.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
    ann.fit(X_train,y_train, batch_size=32, epochs=100)

    y_pred = ann.predict(X_test)

    y_pred = (y_pred > 0.5).astype(int)
    y_test = y_test.reshape(-1)  # ensures proper shape for metrics
    np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

    cm = confusion_matrix(y_test, y_pred)
    print(accuracy_score(y_test, y_pred))



if __name__== "__main__":
    main()
