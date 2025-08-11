import numpy as np
import time

def main():
    vectorToVectorOperation()


def vectorToVectorOperation():
    a = np.array([1, 2, 3, 4])
    b = np.array([-1, -2, 3, 4])
    print("Array a: ",a)
    print("Array b: ",b)
    print(f"Binary operator: {a+b}")

    c = np.array([1,2])
    try:
        d=a+c
    except:
        print("Error Occured")

    print ("5* array a: ",5*a)

    print("Dot Operator: ",np.dot(a,b))

def singleVectorOperation():
    a= np.arange(5)
    print("Full Array: ",a)
    print("Negative Array: ",-a)
    print("Array Sum: ", np.sum(a))
    print("Array Mean: ", np.mean(a))
    print("Array Element Sqaure: ", a**2)

def vectorSlicing():
    a = np.arange(10)
    print(f"a= {a}")

    #access 5 consicutive element (start:stop:step)
    c = a[2:7:1]
    print ("a[2:7:1] = ",c)

    # access 3 elements separated by two
    c =a[2:7:2]
    print("a[2:7:2]",c)

    # access all elements index 3 and above
    c= a[3:]
    print("a[3:]",c)

    # access all elements below index 3
    c = a [:3]
    print("a[:3]",c)

    # access all elements
    c = a[:]
    print("a[:]",c)


def vectorOperations():
    a= np.arange(10)
    print(a)
    print(f"a[2].shape: {a[2].shape} , a[2] = {a[2]}")
    print(f"a[-1]={a[-1]}")                             # Accessing last element using -1 amd can sccess the number from last using negative number
    try:
        c=a[10]
    except:
        print("Error occured")

def vectorCreation():
    a = np.zeros(5)
    print(f"a= {a} , a.shape={a.shape} , a data tyep = {a.dtype}")

    a= np.zeros((5,))
    print(f"a= {a} , a.shape={a.shape} , a data tyep = {a.dtype}")

    a= np.random.random_sample(4)
    print(f"a= {a} , a.shape={a.shape} , a data tyep = {a.dtype}")

    a= np.arange(4.)
    print(f"a= {a} , a.shape={a.shape} , a data tyep = {a.dtype}")

    a= np.random.rand(4)
    print(f"a= {a} , a.shape={a.shape} , a data tyep = {a.dtype}")

    a= np.array([1,2,3,4,5])
    print(f"a= {a} , a.shape={a.shape} , a data tyep = {a.dtype}")

    a= np.array([5.,6,9,10])
    print(f"a= {a} , a.shape={a.shape} , a data tyep = {a.dtype}")


if __name__=="__main__":
    main()