from sklearn.feature_selection import SelectKBest,f_classif
import pandas as pd
from sklearn.datasets import load_iris


def main():
    iris = load_iris()
    print(iris.feature_names)
    X,y = iris.data , iris.target
    selector = SelectKBest(f_classif, k=2)
    X_Selector = selector.fit_transform(X,y)
    selected_indices = selector.get_support(indices=True)
    print(selected_indices)
    selected_feature_names = [iris.feature_names[i] for i in selected_indices]

    print(selected_feature_names)

if __name__== "__main__":
    main()