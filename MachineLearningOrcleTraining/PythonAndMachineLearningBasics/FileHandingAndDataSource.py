import sqlite3
from venv import create

import pandas as pd
from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn import preprocessing
import xlsxwriter
import sqlite3 as sql


def studentPerformance():
    # fetch dataset
    student_performance = fetch_ucirepo(id=320)

    # data (as pandas dataframes)
    X = student_performance.data.features
    y = student_performance.data.targets

    # metadata
    # print(student_performance.metadata)

    # variable information
    # print(student_performance.variables)

    # print(student_performance.data.shape)

    df_president_name = pd.DataFrame({'first': ['George', 'Bill', 'Ronals', 'Jimmy', 'George'],
                                      'last': ['Bush', 'Clinton', 'Regan', 'Carter', 'Washington']})
    # print(df_president_name)
    grouped = df_president_name.groupby('first')
    # print(grouped.get_group('George'))

    df_test_scores = pd.DataFrame({'test1': [95, 84, 73, 88, 82, 61],
                                   'test2': [74, 85, 82, 73, 77, 79]},
                                  index=['jack', 'Lewis', 'patrick', 'rich', 'kelly', 'paula'])
    print(df_test_scores)
    print(standardize_tests(df_test_scores))
    print(standardize_tests_score(df_test_scores))

def standardize_tests(test):
    return(test-test.mean())/test.std()


def standardize_tests_score(test):
    return test.apply(standardize_tests)

def dataLabeling():
    input_labels = ['justice', 'patience', 'justice', 'truth', 'patience', 'loyalty', 'perseverance']
    encoder = preprocessing.LabelEncoder()
    encoder.fit(input_labels)
    test_labels = ['truth', 'justice', 'patience', 'loyalty']
    encoded_values = encoder.transform(test_labels)
    print(test_labels)
    encoded_values = [2, 0, 4, 3]
    decoded_list = encoder.inverse_transform(encoded_values)
    print(encoded_values)
    print(list(decoded_list))

def dataFrameSample():
    d = {'s1': [100, 90, np.nan, 95], 's2': [30, 45, 56, np.nan], 's3': [np.nan, 40, 50, 60]}
    df = pd.DataFrame(d)
    print(df)
    print(df.isnull())
    print(df.fillna('BAC'))

def excelReaderAndWriter():
    wb = xlsxwriter.Workbook('../../TestData/mybook.xlsx')
    worksheet = wb.add_worksheet()
    chart = wb.add_chart({'type': 'line'})

    data = [
        [10, 20, 30, 40, 50],
        [20, 40, 60, 80, 100],
        [30, 60, 90, 120, 150],
    ]
    worksheet.write_column('A1', data[0])
    worksheet.write_column('B1', data[1])
    worksheet.write_column('C1', data[2])

    chart.add_series({'values': '=Sheet1!$A$1:$A$5'})
    chart.add_series({'values': '=Sheet1!$B$1:$B$5'})
    chart.add_series({'values': '=Sheet1!$C$1:$C$5'})

    worksheet.insert_chart('B7', chart)

    wb.close()

def sqlLiteImplementaion():
    create_table = """ create table student_score 
    (id INTEGER,name VARCHAR(20),math REAL,science REAL);"""
    executeSQL = sqlite3.connect(':memory:')
    executeSQL.execute(create_table)
    insertSQL = [(10, 'lakshmi', 89, 92), (20, 'padmasree', 89, 98), (30, 'sree', 78, 98)]
    insert_statement = "Insert into student_score values(?,?,?,?)"
    executeSQL.executemany(insert_statement, insertSQL)
    executeSQL.commit()
    df_student_rec = pd.DataFrame(executeSQL.execute('select * from student_score').fetchall())
    print(df_student_rec)

def main():
    sqlLiteImplementaion()

if __name__== "__main__":
    main()