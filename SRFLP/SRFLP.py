from dis import dis
import itertools as it
import multiprocessing as mp
from multiprocessing.sharedctypes import Value
from xml.dom import minicompat
import numpy as np
import os

def LoadFile(filename):
    dataset = {}
    file = open(filename, 'r')

    # Read 1. line
    dataset['dim'] = (int(file.readline()))    
    
    # Read 2. line
    line = file.readline()
    line.strip()
    line = line.split()
    dataset['widths'] = [int(w) for w in line]

    # Read rest of the file
    distances = []
    while True:
        line = file.readline()
        line = line.split()
        distances.append([int(d) for d in line])
        if not line:
            dataset['dist'] = distances
            break
    
    file.close()
    return dataset

# Generate permutations with given prefix
def PrefixPermutations(n, s):
    numbers_to_permutate = [*range(n)]
    numbers_to_permutate.remove(s)
    permutations = []
    for permutation in it.permutations(numbers_to_permutate):
        p = list(permutation)
        p.insert(0, s)
        permutations.append(p)

    return permutations

def SRFLP(permutation, dataset):
    result = 0.0
    n = dataset['dim']
    for i in range(0, n-1):
        for j in range(i+1, n):
            a = min(permutation[i], permutation[j])
            b = max(permutation[i], permutation[j])
            dist = dataset['dist'][a][b]
            width = Distance(permutation, i,j,dataset['widths'])
            print(f"{a} {b} - {dist} * {width}")
            result += ( dist * width )
    return result

# SRFLP function usin branch & bound
def SRFLP_bnb(permutation, dataset):
    global minimum_shared
    result = 0
    n = dataset['dim']
    for i in range(0, n-1):
        for j in range(i+1, n):
            a = min(permutation[i], permutation[j])
            b = max(permutation[i], permutation[j])
            result += ( dataset['dist'][a][b] * Distance(permutation, i, j, dataset['widths']) )

            # Stop if result is already too big
            if result >= minimum_shared.value:
                return False, i+1

    return True, result


def Distance(permutation, i, j, widths):
    p1 = min(permutation[i],permutation[j])
    p2 = max(permutation[i],permutation[j])
    
    frac = (widths[p1] + widths[p2]) / 2.0    # Fraction part

    sum = 0                                 # Sum part
    for k in range(i+1,j):
        sum += widths[k]

    return frac + sum


def ThreadTest(n):
    global shared_n
    print(f"Thread {n} starting with {shared_n}")
    shared_n = shared_n + 100

def ThreadEvaluate(dataset, prefix):
    #print(prefix)
    global minimum_shared
    n = dataset['dim']
    #print(n)
    permutations = PrefixPermutations(n, prefix)    # Generate all permutations with prefix
    
    skip = False
    skipIndex, skipValue = -1, -1

    # Iterate over permutations
    for permutation in permutations:
        # Check if this permutation should be skipped
        if(skip):
            if(permutation[skipIndex] != skipValue):
                skip = False

        # Evaluate permitation if not skipped
        if(skip == False):
            newFound, res = SRFLP_bnb(permutation, dataset)

            if newFound:
                with minimum_shared.get_lock():
                    if res < minimum_shared.value:
                        minimum_shared.value = res
                        minimum_perm = permutation
                        print(f"New mnimum found on {minimum_perm} - {res}")
            else:
                skipIndex = res
                skipValue = permutation[skipIndex]
                skip = True

    return minimum_shared.value

# Initialize global variable
def init_workers(args):
    global minimum_shared
    minimum_shared = args

global minimum_perm
minimum_perm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if __name__ == '__main__':
    filename = "Y-10_t.txt"
    data = LoadFile(filename)       # Read data from file

    shared = Value('f', 100000)
    pool = mp.Pool(initializer=init_workers, initargs=(shared,))    # Create thread pool

    #print(minimum_perm)
  
    # Fill argument list
    inputs = []
    for i in range(data['dim']):
        inputs.append([data, i])

    results = pool.starmap(ThreadEvaluate, inputs)
    print(f"Best solution: {results[0]} - {minimum_perm}")

    #pool = mp.Pool(initializer=init_workers, initargs=(10000,)) 
    #inputs = [i for i in range(0,10)]
    #res = pool.map(ThreadTest, inputs)
    #print(res)

    #permutation = [0, 4, 1, 9, 6, 3, 7, 2, 5, 8]
    #print(SRFLP(permutation, data))
