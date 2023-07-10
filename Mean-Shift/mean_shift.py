import csv
import math
import numpy as np
import multiprocessing as mp

def OpenFile(filename):
    result = []
    labels = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count >= 1 and line_count % 3 == 0:
                labels.append(int(row[0]))                  # Append label
                row.pop(0)                                  # Remove label
                row = np.array([eval(x) for x in row])      # String to int
                result.append(row)                          # Append data
                line_count += 1

            else:
                line_count +=1  # Don't read the first row

    return result, labels 

# Euclidian distance
def Distance(a, b):
    dist = np.linalg.norm(a-b)
    return dist

# Get surrounding points of some x
def N(x, dataset, window):
    result = []
    for point in dataset:
        d = Distance(x, point)
        if d <= window:             # Only append points in given radius
            result.append(point)
    return result

# Gauss kernel function
def Kernel(a, sigma):
    fraction = (1/2) * (a/sigma**2)
    e = np.exp(-fraction)
    return e

def Shift(x, neighbors, window):
    sum_up = 0      # Fraction up part
    sum_down = 0    # Fraction down part

    for x_i in neighbors:
        sum_up += ( Kernel(x_i - x, window) * x_i )
        sum_down += Kernel(x_i - x, window)
    
    shift = sum_up/sum_down
    return shift

def GetCentroid(point, centroids, maxCentroidDistance):
    row = -1
    for centroid in centroids:
        row += 1    # Centroid number
        if Distance(point, centroid) <= maxCentroidDistance:
            return True, row
    
    # Centroid was not found
    return False, point

# Sequential Mean Shift
def MeanShift_seq(dataset, window, stopDistance):
    centroids = []              # Positions of centroids
    point_assigments = []       # Assigments to centroides

    # Main loop over points
    for x in dataset:
        move = True
        
        # Moving one point 
        while(move):
            neighbors = N(x, dataset, window)           # Neighbors in givern radius
            newPosition = Shift(x, neighbors, window)   # New position
            d = Distance(x, newPosition)

            if d <= stopDistance:
                move = False
                # Assign centroid
                centroidExist, centroidPosition = GetCentroid(newPosition, centroids, 5.0)

                if centroidExist:
                    # Assign ID of existing centroid
                    point_assigments.append(centroidPosition)
                else:
                    # Create new centroid and assign ID
                    centroids.append(centroidPosition)
                    point_assigments.append(len(centroids)-1)   # Assign new ID of created centroid

            else:
                x = newPosition     # Move x to a new position

# TODO: Parallel Mean Shift

def MeanShift_parallel(dataset, window, stopDistance):
    centroids = []              # Positions of centroids
    point_assigments = []       # Assigments to centroides

    # Main loop over points
    for x in dataset:
        move = True
        
        # Moving one point 
        while(move):
            neighbors = N(x, dataset, window)           # Neighbors in givern radius
            newPosition = Shift(x, neighbors, window)   # New position
            d = Distance(x, newPosition)

            if d <= stopDistance:
                move = False
                # Assign centroid
                centroidExist, centroidPosition = GetCentroid(newPosition, centroids, 5.0)

                if centroidExist:
                    # Assign ID of existing centroid
                    point_assigments.append(centroidPosition)
                else:
                    # Create new centroid and assign ID
                    centroids.append(centroidPosition)
                    point_assigments.append(len(centroids)-1)   # Assign new ID of created centroid

            else:
                x = newPosition     # Move x to a new position

def do_stuff(x, sharedlist):
    sharedlist.append(x)
    print(f"Thread {x}: {sharedlist}")

# Get argument list for threads
def GetThreadArguments(dataset, window, stopDistance, rows, nThreads):
    result = []
    rowsInThread = rows // nThreads     # How many rows one thread should do

    for i in range(0,nThreads-1):
        dataInThread = [dataset[j] for j in range(i*rowsInThread, i*rowsInThread + rowsInThread)]
        result.append( (dataInThread, window, stopDistance) )

    # Last thread needs to do the rest
    dataInThread = [dataset[j] for j in range((nThreads-1) * rowsInThread, rows)]
    result.append( (dataInThread, window, stopDistance) )

    return result

# Main function
def main():
    #filename_train = "mnist_dataset/mnist_rain.csv"
    filename_test = "mnist_dataset/mnist_test.csv"

    # Reading dataset
    print("Loading dataset...")
    #dataset, labels = OpenFile(filename_test)
    #rows = len(dataset)
    #print(f"{rows} rows loaded.")

    #MeanShift_seq(dataset, 1800, 20)

    # Testing the parallel list sharing

    data = [x for x in range(0,20)]
    dataArgs = GetThreadArguments(data, 5, 5, len(data), 6)

    x = 5
    #pool=mp.Pool(processes=6)
    #manager=mp.Manager()
    #sharedlist=manager.list()
    #tasks = [(x,sharedlist) for x in range(0,6)]
    #pool.starmap(do_stuff, tasks)
    #pool.close()
    #print(sharedlist)

if __name__ == "__main__":
    main()