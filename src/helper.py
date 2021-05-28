import numpy as np

import matplotlib.pyplot as plt

#from numba import njit
#import numba

#import math

"""
arg:
    n: Number of tasks
    maxDuration: The duration are in [1, maxDuration[

return random duration array drawn according to an uniform distribution
"""
def generateRandomDurations(n, maxDuration=3):

    return np.random.randint(1, maxDuration, size=n)


"""
arg:
    n: Number of tasks
    exposant: exposant for the pareto distribution

return random duration array drawn according to a pareto distribution
"""
def generateRandomParetoDurations(n, exposant=1.1):

    return np.random.pareto(exposant, size=n)


"""
arg:
    n: Number of tasks
    exposant: exposant for the pareto distribution

return random arrives array drawn according to a pareto distribution
"""
def generateRandomArrives(n, exposant=0.92):

    return np.random.pareto(exposant, size=n)

"""
Same but with integer using an uniform distribution
"""
def generateRandomArrivesInteger(n, maxTime=10):

    return np.random.randint(0, maxTime, size=n)



"""
arg:
    n: Number of tasks
    durations: durations array
    variation: the error

return random integer prediction array drawn according to an uniform distribution
"""
def generatePredictionsOnDurations(n, durations, variation=2):

    ret = durations + np.random.randint(-variation, variation+1, size=n)
    
    return np.where(ret < 1, 1, ret)

"""
arg:
    n: Number of tasks
    durations: durations array
    sigma: the error

return random prediction array drawn according to a normal distribution of mu 0
"""
def generateNormalPredictionsOnDurations(n, durations, sigma=0.1):

    ret = durations + np.random.normal(0, sigma, n)
    
    return np.where(ret < 0, 0, ret)



"""
arg:
    n: Number of tasks

return a scheduling array
"""
def genRandomScheduling(n):

    rez = np.arange(n)
    np.random.shuffle(rez)

    return rez




"""
OPT strategie when knowing the duration

arg:
    durations: task duration array

return a scheduling array
"""
def genSortedScheduling(durations):

    return durations.argsort()





def testScheduling(durations, scheduling):

    acc = 0
    score = 0
    for sche in scheduling:

        acc += durations[sche]

        score += acc

    return score

#@njit(cache=True)
def sortedDurationWithArrive(durations, arrive):
    timeSpent = np.zeros(durations.size, dtype=np.float64)
    states = np.zeros(durations.size, dtype=np.int8)

    totalTime = 0
    score = 0

    indexes = np.argwhere(arrive == 0).ravel()

    iteration = 0

    #Note: isclose because of python precision
    while not (np.abs(timeSpent - durations) < 1.e-8).all():

        iteration += 1

        notFinished = ~(np.abs(timeSpent[indexes] - durations[indexes]) < 1.e-8)

        miniValue = np.inf

        if notFinished.sum() > 0:

            mini = (durations[indexes][notFinished] - timeSpent[indexes][notFinished]).argmin()

            miniIndex = indexes[notFinished][mini]

            miniValue = durations[miniIndex] - timeSpent[miniIndex]


        mask = np.ones(arrive.shape, np.bool_)
        mask[indexes] = False

        mask = np.argwhere(mask).ravel()

        timeToSpent = min((arrive[mask] - totalTime).min(), miniValue) if mask.size > 0 else miniValue


        totalTime += timeToSpent

        #Add only if there is a task
        if notFinished.sum() > 0:
            timeSpent[miniIndex] += timeToSpent

        done = np.argwhere(np.abs(timeSpent - durations) < 1.e-8).ravel()

        score += (states[done] != 1).sum() * totalTime

        states[done] = 1


        close = np.argwhere(np.abs(arrive[mask] - totalTime) < 1.e-8).ravel()

        indexes = np.concatenate( (indexes, mask[close]) )
        indexes.sort()

    return score

"""
def sortedDurationWithArrive(durations, arrive):
    timeSpent = np.zeros(durations.size, dtype=np.float64)
    states = np.zeros(durations.size, dtype=np.int8)

    totalTime = 0
    score = 0

    indexes = np.argwhere(arrive == 0).ravel()

    iteration = 0
    
    scoreArr = []
    totalTimeArr = []
    
    tacheTimeArr = []

    #Note: isclose because of python precision
    while not (np.abs(timeSpent - durations) < 1.e-8).all():

        iteration += 1

        notFinished = ~(np.abs(timeSpent[indexes] - durations[indexes]) < 1.e-8)

        miniValue = np.inf

        if notFinished.sum() > 0:

            mini = (durations[indexes][notFinished] - timeSpent[indexes][notFinished]).argmin()

            miniIndex = indexes[notFinished][mini]

            miniValue = durations[miniIndex] - timeSpent[miniIndex]


        mask = np.ones(arrive.shape, np.bool_)
        mask[indexes] = False

        mask = np.argwhere(mask).ravel()

        timeToSpent = min((arrive[mask] - totalTime).min(), miniValue) if mask.size > 0 else miniValue


        totalTime += timeToSpent

        #Add only if there is a task
        if notFinished.sum() > 0:
            timeSpent[miniIndex] += timeToSpent

        done = np.argwhere(np.abs(timeSpent - durations) < 1.e-8).ravel()

        score += (states[done] != 1).sum() * totalTime

        states[done] = 1


        #Verbose to print into a latex format
        print("\\noindent Itération: "+str(iteration)+"\\\\\nTemps attribué pour chaque tache disponible:\\\\")
        for ind, task in enumerate(indexes):
            if ind+1 == len(indexes):
                if task == miniIndex and notFinished.sum() > 0:
                    print(str(task)+": "+str(round(timeToSpent, 3))+"\\\\")
                else:
                    print(str(task)+": "+str(0)+"\\\\")
            else:
                if task == miniIndex and notFinished.sum() > 0:
                    print(str(task)+": "+str(round(timeToSpent, 3))+", ", end="")
                else:
                    print(str(task)+": "+str(0)+", ", end="")


        tmpArr = []
        for task, time in enumerate(timeSpent):
            if task+1 == len(timeSpent):
                tmpArr += [round(time, 3)]
            else:
                tmpArr += [round(time, 3)]
        
        tacheTimeArr += [tmpArr]

        totalTimeArr += [round(totalTime, 3)]

        #to put in a array
        scoreArr += [round(score, 3)]

        print()


        close = np.argwhere(np.abs(arrive[mask] - totalTime) < 1.e-8).ravel()

        indexes = np.concatenate( (indexes, mask[close]) )
        indexes.sort()

    return score, totalTimeArr, scoreArr, tacheTimeArr


"""



"""
Execute an algorithm on an instance "durations".

The algorithm take the number of task, the predictions and a state array
(which tell if a task if finished or not)
and should return an array which contain
the fraction of the computer allocated to the task at index i.
It sum should be equal to 1.

"""
#@njit(cache=True)
def executeAlgorithm(durations, predictions, algorithm, arrive=np.empty(0)):

    if arrive.size == 0:
        arrive = np.zeros(durations.size)

    currentIndexes = np.argwhere(arrive == 0).ravel()

    n = len(durations)
    states = np.zeros(n, dtype=np.int8)
    timeSpent = np.zeros(n, dtype=np.float64)

    iteration = 0
    score = 0

    totalTime = 0


    while states.sum() < n:

        iteration += 1

        smallestCoef = np.inf

        #This test make sure there is a task available
        if np.argwhere(states[currentIndexes] < 1).ravel().size > 0:

            allocation = algorithm( len(currentIndexes), predictions[currentIndexes], states[currentIndexes] )

            # if not math.isclose(allocation.sum(), 1):
            #     print("Allocation sum not equal to 1, cheating")
            #     return -1

            #smallestCoef = np.inf
            for alloc, (alreadySpent, dura) in zip(allocation, zip(timeSpent[currentIndexes], durations[currentIndexes])):

                if alloc:

                    rez = (dura - alreadySpent) / alloc

                    if rez < smallestCoef:
                        smallestCoef = rez

        else:
            allocation = np.zeros(currentIndexes.size)


        mask = np.ones(arrive.shape, np.bool_)
        mask[currentIndexes] = False

        mask = np.argwhere(mask).ravel()


        #Take the min time between the next finished task with the algorithm allocation or the next new task
        smallestCoef = min((arrive[mask] - totalTime).min(), smallestCoef) if mask.size > 0 else smallestCoef



        totalTime += smallestCoef

        timeSpent[currentIndexes] += allocation * smallestCoef

        #Note: we're using here < 1.e-8 instead of using numpy.isclose because of the use of numba.
        done = np.argwhere(np.abs(timeSpent - durations) < 1.e-8).ravel()

        score += (states[done] != 1).sum() * totalTime


        states[done] = 1

        #Verify if we add task
        #np.abs(arrive[mask] - totalTime) < 1.e-8
        close = np.argwhere(np.abs(arrive[mask] - totalTime) < 1.e-8).ravel()

        currentIndexes = np.concatenate( (currentIndexes, mask[close]) )
        currentIndexes.sort()

    return score

#To return more data
"""
def executeAlgorithm(durations, predictions, algorithm, arrive=np.empty(0)):

    if arrive.size == 0:
        arrive = np.zeros(durations.size)

    currentIndexes = np.argwhere(arrive == 0).ravel()

    n = len(durations)
    states = np.zeros(n, dtype=np.int8)
    timeSpent = np.zeros(n, dtype=np.float64)

    iteration = 0
    score = 0

    totalTime = 0
    
    scoreArr = []
    totalTimeArr = []
    
    tacheTimeArr = []

    while states.sum() < n:

        iteration += 1

        smallestCoef = np.inf

        #This test make sure there is a task available
        if np.argwhere(states[currentIndexes] < 1).ravel().size > 0:

            allocation = algorithm( len(currentIndexes), predictions[currentIndexes], states[currentIndexes] )

            # if not math.isclose(allocation.sum(), 1):
            #     print("Allocation sum not equal to 1, cheating")
            #     return -1

            #smallestCoef = np.inf
            for alloc, (alreadySpent, dura) in zip(allocation, zip(timeSpent[currentIndexes], durations[currentIndexes])):

                if alloc:

                    rez = (dura - alreadySpent) / alloc

                    if rez < smallestCoef:
                        smallestCoef = rez

        else:
            allocation = np.zeros(currentIndexes.size)


        mask = np.ones(arrive.shape, np.bool_)
        mask[currentIndexes] = False

        mask = np.argwhere(mask).ravel()


        #Take the min time between the next finished task with the algorithm allocation or the next new task
        smallestCoef = min((arrive[mask] - totalTime).min(), smallestCoef) if mask.size > 0 else smallestCoef



        totalTime += smallestCoef

        timeSpent[currentIndexes] += allocation * smallestCoef

        #Note: we're using here < 1.e-8 instead of using numpy.isclose because of the use of numba.
        done = np.argwhere(np.abs(timeSpent - durations) < 1.e-8).ravel()

        score += (states[done] != 1).sum() * totalTime


        states[done] = 1

        tmpArr = []
        for task, time in enumerate(timeSpent):
            if task+1 == len(timeSpent):
                tmpArr += [round(time, 3)]
            else:
                tmpArr += [round(time, 3)]
        
        tacheTimeArr += [tmpArr]

        totalTimeArr += [round(totalTime, 3)]

        #to put in a array
        scoreArr += [round(score, 3)]

        #Verify if we add task
        #np.abs(arrive[mask] - totalTime) < 1.e-8
        close = np.argwhere(np.abs(arrive[mask] - totalTime) < 1.e-8).ravel()

        currentIndexes = np.concatenate( (currentIndexes, mask[close]) )
        currentIndexes.sort()

    return score, totalTimeArr, scoreArr, tacheTimeArr


"""


def rapport(optimum, solution):
    return solution/optimum

"""
    
#Debug
import algorithm
if __name__ == "__main__":
    n = 10
    error = 0.2
    durations    = generateRandomParetoDurations(n)
    predictions  = generateNormalPredictionsOnDurations(n, durations, sigma=error)
    arrive = generateRandomArrives(n, exposant=0.92)
    
    

    durations = np.array([ 4.36618386,0.35502595,2.34828367,0.3507639,1.03361353,7.22382529,0.28340898,3.39257255,0.24312944,0.36299601 ])
    predictions = np.array( [4.21771975, 0.43932975, 2.51858838, 0.40686916, 1.17715815, 20.78895229, 0.17439453, 3.35221127, 0.44273154, 0.32329664])
    arrive = np.array([3.78971342, 6.011637, 1.41528382, 0.22950969, 0.29262065, 1.41659193, 0.47066157, 1.04768535, 1.34561503, 0.80059455])

    #score, totalTimeArr, scoreArr, tacheTimeArr = sortedDurationWithArrive(durations, arrive=arrive)
    #score, totalTimeArr, scoreArr, tacheTimeArr = executeAlgorithm(np.array([3, 2, 2, 5, 2]), np.array([3, 3, 0, 4, 1]), algorithm.shortestPred, arrive=np.array([3, 0, 0, 4, 1]))
    score, totalTimeArr, scoreArr, tacheTimeArr = executeAlgorithm(durations, predictions, algorithm.shortestPred, arrive=arrive)
    
    tacheTimeArr = np.array(tacheTimeArr).T
    tickPos = np.array([x for x in range(len(totalTimeArr))])
    
    stepSizeArr = [totalTimeArr[0]]
    
    for i in range(1,len(totalTimeArr)):
        stepSizeArr += [np.abs(totalTimeArr[i-1]-totalTimeArr[i])]
        
    totalTimeArr = np.around(totalTimeArr,2)
    
    plt.figure(figsize=(8, 10), dpi=80)
    plt.suptitle("shortest pred with arrive", fontsize=12)
    plt.subplot(3, 1, 1)
    for i in range(len(durations)):
        plt.plot(tacheTimeArr[i])
    plt.xlabel("time")
    plt.xticks(tickPos,totalTimeArr)
    plt.title("task usage evolution")
    plt.ylabel("dedicated execution time")
    plt.subplot(3, 1, 2)
    plt.plot(scoreArr)
    plt.title("score evolution")
    plt.xlabel("time")
    plt.xticks(tickPos,totalTimeArr)
    plt.ylabel("score")
    plt.subplot(3, 1, 3)
    plt.plot(stepSizeArr)
    plt.title("time step")
    plt.ylabel("time step size")
    plt.xlabel("step")
    plt.xticks(tickPos,tickPos)
    plt.tight_layout()
    plt.show()


"""