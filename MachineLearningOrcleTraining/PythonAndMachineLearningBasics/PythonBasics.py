import platform
import sys

# take input marks and grade them :
# A => 90 and above
# B => 75 - 90
# C => 55 - 75
# D => 35 - 55
# F => <35

def main():
    lambdaFuntion()


def lambdaFuntion():
    list2 = [20, 50, 39, 102, 339, 220]
    mylist = map(lambda x: x * 2, list2)
    print(list(mylist))

    list1 = [20, 50, 39, 102, 339, 220]
    result = list(filter(lambda x: (x % 5 == 0), list1))
    print(result)

    average = lambda x, y: (x + y) / 2
    print(average(4, 3))


def dataStructure():
    """
    tupels()
    List []
    set {}
    """
    # tup1 = (12,'vivek', 45.3, (3,2), True)
    # print(tup1)
    # list1 = ['lakshmi',101,25.6,True,11]
    # list1[0] = 'vivek'
    # print(list1)
    # list1.append(5)
    # print(list1)
    # list1.remove(11)
    # print(list1)

    # input1 = {'Brazil': ['Brasilia', 17.3], 'Russia': ['Moscow', 3.28], 'India': ['Delhi', 4.5],
    #           'China': ['Beijing', 9.6], 'South Africa': ['Pretoria', 1.22]}
    # print(input1)
    # print(input1.keys())
    # print(input1.values())

    # str = "learning Datascience"
    # # displaying whole
    # print(str)
    # # displaying first character of
    # print(str[0])
    # # displaying third character
    # print(str[2])
    # # displaying the last characterof the
    # print(str[-1])
    # # displaying the second last
    # print(str[-2])

    # str = "Hello"
    # str1 = " world"
    # print(str * 3)  # prints HelloHelloHello
    # print(str + str1)  # prints Hello world
    # print(str[4])  # prints o
    # print(str[2:4]);  # prints ll
    # print('w' in str)  # prints false as w is not present in str
    # print('wo' not in str1)  # prints false as wo is present in str1.
    # print(r'C://python')  # prints C://python37 as it is written
    # print("The string str : %s" % (str))  # prints The string str : Hello

    str = "learning python"
    str2 = "Oracle"
    print(str + str2)
    print(str2 * 5)
    print(str[2:6])
    print("Sigma" in str)
    print("Sigma" not in str)
    print(r'c://Python')
    print("String format is %s" % str)
    print(len(str))
    print(str.capitalize())
    print(str2.isupper())
    print(str.istitle())
    print("{} {}".format(str, str2))
    l = 10
    b = 20
    print("{} {}".format(l, b))
    print(str.count('l'))
    print(str.find('g'))
    print(str.split(" "))


def grade():
    marks = float(input("Enter your marks: "))

    if marks >= 90:
        grade = 'A'
    elif marks >= 75:
        grade = 'B'
    elif marks >= 55:
        grade = 'C'
    elif marks >= 35:
        grade = 'D'
    else:
        grade = 'F'

    print(f"Your grade is: {grade}")

if __name__== "__main__":
    main()