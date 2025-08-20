import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import r2_score
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from  sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

def main():
    regeressionAlgo_heightWeightData()

def regeressionAlgo_heightWeightData():
    df = pd.read_csv("../TestData/height-weight.csv")
    # print(df)
    # plt.scatter(df['Height'],df['Weight'])
    # plt.show()
    X= df['Weight'].values
    y = df['Height'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = np.array(X_train).reshape(-1,1)
    X_test = np.array(X_test).reshape(-1, 1)

    print(X_train)
    print(X_test)

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    print(X_train)
    print(X_test)

    # plt.scatter(X_train, df['Weight'])
    # plt.show()

    regressor = LinearRegression().fit(X_train, y_train)

    y_pridict = regressor.predict(X_test)
    print(y_pridict)

    model = sm.OLS(y_train, X_train).fit()
    Predictions = model.predict(X_test)
    print(model.summary())

    plt.scatter(X_test, y_test)
    plt.show()

    # r2 = r2_score(y_test, y_pridict) * 100
    # print('R² Score of our model is equal to ' + str(round(r2, 2)) + ' %.')
    #
    # print('R Score of our model is equal to ' + str(round(model.score(X_test, y_test) * 100, 2)) + ' %.')
    # plt.scatter(X_train,y_train)
    # plt.show()


def regeressionAlgo_SalaryData():
    salarydata = pd.read_csv("../TestData/Salary_Data.csv")
    # print(salarydata)
    X = salarydata.iloc[:,:1].values
    y= salarydata.iloc[:,1:].values
    # print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
    model = LinearRegression().fit(X_train,y_train)

    y_pridict = model.predict(X_test)
    print(y_pridict)

    r2 = r2_score(y_test, y_pridict) * 100
    print('R² Score of our model is equal to ' + str(round(r2, 2)) + ' %.')

    print('R Score of our model is equal to ' + str(round(model.score(X_test,y_test)*100, 2)) + ' %.')

    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, model.predict(X_train), color='blue')
    plt.title('Salary vs Experience (Training set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()


if __name__== "__main__":
    main()