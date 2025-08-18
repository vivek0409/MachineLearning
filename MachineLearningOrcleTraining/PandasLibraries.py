import pandas as pd

def main():
    sample_data = pd.read_csv("../TestData/sample_data.csv")
    print(sample_data)


if __name__== "__main__":
    main()