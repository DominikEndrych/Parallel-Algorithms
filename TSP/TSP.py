
from dis import dis
import itertools as it        # req. step 2
import multiprocessing as mp  # req. step 3
import numpy as np
import os

################################ step 1
def load(fname):
  f = open(fname)

  distances = []

  for line in f:
    line = line.strip()
    one_line = line.split("\t")
    #distances.append(one_line)
    distances.append([int(d) for d in one_line])

  n = int(distances[0][0])
  distances.pop(0)

  return distances, n

def evaluate(distances, perm, n):
  sum = 0
  for i in range(n - 1):
    sum += int(distances[int(perm[i])][int(perm[i+1])])

  sum += int(distances[int(perm[n-1])][int(perm[0])])    # Final distance
  return sum

def run_step1():
  distances, n = load('ulysses16.txt')
  perm = list(range(16))
  print(evaluate(distances, perm, n))

#run_step1()


############################### step 2
def perms(n):
    permutations = []
    for p in it.permutations(range(n)):
        permutations.append(p)
    return permutations
    

def run_step2():
    n = 4
    perms(n)

############################### step 3
def perms_prefix(n, s):
    numbers_to_permutate = [*range(n)]
    numbers_to_permutate.remove(s)
    permutations = []
    for permutation in it.permutations(numbers_to_permutate):
        p = list(permutation)
        p.insert(0, s)
        permutations.append(p)

    return permutations
  

def run_step3():
    n = 5

    pool = mp.Pool()
  
    # Fill argument list
    input_p = []
    for i in range(n):
        input_p.append((n,i))

    print(input_p)
    permutations = pool.starmap(perms_prefix, input_p)

    print(permutations)

    #for i in range(n):
    #    p = perms_prefix(n, i)
    #    print(p)

#branch and bound
def new_evaluate(permutation, distances, n, currentMax):
    sum = 0

    # Trip around permutation
    for i in range(n-1):
        sum += distances[permutation[i]][permutation[i+1]]
        if sum >= currentMax:
            return False, i+1

    # Trip from last to first
    sum += distances[permutation[n-1]][permutation[0]]
    if sum >= currentMax:
        return False, n-1

    return True, sum

def bnb_evaluate():
    shortest = 99999999999
    distances, n = load('ulysses16.txt')
    #permutations = perms(n)

    skip = False
    skipValue = -1
    skipIndex = -1
    for p in it.permutations(range(9)):
        #print(p)
        if(skip):
            if(p[skipIndex] != skipValue): skip = False
        else:
            newFound, res = new_evaluate(p, distances, 9, shortest)
            if(newFound):
                shortest = res
            else:
                skip = True
                skipIndex = res
                skipValue = p[skipIndex]

    return shortest

def bnb_evaluate_serial(n, s, distances):
    shortest = 99999999999
    #distances, n = load('ulysses16.txt')
    #permutations = perms(n)

    skip = False
    skipValue = -1
    skipIndex = -1
    for p in perms_prefix(n, s):
        #print(p)
        if(skip):
            if(p[skipIndex] != skipValue): skip = False
        else:
            newFound, res = new_evaluate(p, distances, n, shortest)
            if(newFound):
                shortest = res
                shortest_perm = p
            else:
                skip = True
                skipIndex = res
                skipValue = p[skipIndex]

    return shortest, shortest_perm

def run_step4():
    result = bnb_evaluate()
    print(result)

def run_step5():
    distances, n = load('ulysses16.txt')
    n = 8

    pool = mp.Pool()
  
    # Fill argument list
    input_p = []
    for i in range(n):
        input_p.append((n,i,distances))

    #print(input_p)
    results = pool.starmap(bnb_evaluate_serial, input_p)

    print(results)

if __name__ == '__main__':
    #run_step2()
    #perms_prefix(5, 0)
    #run_step3()
    
    #run_step4()        # bnb serial
    run_step5()         # bnb paralel

