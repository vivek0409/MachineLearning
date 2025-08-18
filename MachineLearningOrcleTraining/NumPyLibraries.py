import numpy as np


def main():
    distance = []
    time = []
    np_distance = np.array(distance)
    np_time = np.array(time)

    first_trial_cyclist = [10, 15, 17, 26]
    second_trail_cyclist = [12, 11, 21, 24]
    np_first = np.array(first_trial_cyclist)
    np_second = np.array(second_trail_cyclist)
    np_total = np_first + np_second
    # print(np_total)

    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    # print(x+y)
    # print(np.add(x,y))
    # print(x-y)
    # print(np.subtract(x,y))
    # print(x*y)
    # print(np.multiply(x,y))
    # print(x/y)
    # print(np.divide(x,y))
    # print(x.sum())
    # print(y.cumsum())
    # print(x.min(), x.max(), x.mean())

    ## shape of an array using shape manipulation functions
    ##flatten,split,stack,reshape,resize

    result = np.array([[10, 15, 17, 26, 23, 20], [12, 11, 21, 24, 34, 23]])
    print(result.shape)
    print(result.reshape(3,4))
    print(np.hsplit(result,2))
    print(np.vsplit(result,2))

if __name__== "__main__":
    main()