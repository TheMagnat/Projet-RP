
import numpy as np

from numba import njit
import numba

import math

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

@njit(cache=True)
def sortedDurationWithArrive(durations, arrive):
	timeSpent = np.zeros(durations.size, dtype=np.float64)
	states = np.zeros(durations.size, dtype=np.int8)

	totalTime = 0
	score = 0

	indexes = np.argwhere(arrive == 0).ravel()

	#Note: isclose because of python precision
	while not (np.abs(timeSpent - durations) < 1.e-8).all():

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
Execute an algorithm on an instance "durations".

The algorithm take the number of task, the predictions and a state array
(which tell if a task if finished or not)
and should return an array which contain
the fraction of the computer allocated to the task at index i.
It sum should be equal to 1.

"""
@njit(cache=True)
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
			# 	print("Allocation sum not equal to 1, cheating")
			# 	return -1

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

		done = np.argwhere(np.abs(timeSpent - durations) < 1.e-8).ravel()

		score += (states[done] != 1).sum() * totalTime


		states[done] = 1


		#Verify if we add task
		#np.abs(arrive[mask] - totalTime) < 1.e-8
		close = np.argwhere(np.abs(arrive[mask] - totalTime) < 1.e-8).ravel()

		currentIndexes = np.concatenate( (currentIndexes, mask[close]) )
		currentIndexes.sort()

	return score

#Olf version without arrive
"""
def executeAlgorithm(durations, predictions, algorithm):

	n = len(durations)
	states = np.zeros(n, dtype=np.int)
	timeSpent = np.zeros(n, dtype=np.float64)


	iteration = 0
	score = 0

	totalTime = 0

	while states.sum() < n:

		iteration += 1

		allocation = algorithm(n, predictions, states.copy())

		if not np.isclose(allocation.sum(), 1):
			print("Allocation sum not equal to 1, cheating")
			return -1

		smallestCoef = np.inf
		for alloc, (alreadySpent, dura) in zip(allocation, zip(timeSpent, durations)):

			if alloc:

				rez = np.linalg.solve(a=[[ alloc ]], b=[dura - alreadySpent])

				if rez[0] < smallestCoef:
					smallestCoef = rez[0]


		totalTime += smallestCoef

		timeSpent += allocation * smallestCoef

		done = np.argwhere(np.isclose(timeSpent, durations)).ravel()

		score += (states[done] != 1).sum() * totalTime

		states[done] = 1


	return score
"""


#One by one version, not working with predRoundRobin
# def executeAlgorithm(durations, predictions, algorithm):

# 	n = len(durations)
# 	states = np.zeros(n, dtype=np.int)
# 	timeSpent = np.zeros(n, dtype=np.float64)


# 	iteration = 0
# 	score = 0
# 	while states.sum() < n:

# 		iteration += 1

# 		allocation = algorithm(n, predictions, states.copy())

# 		if allocation.sum() > 1:
# 			print("Allocation > 1, cheating")

# 		timeSpent += allocation

# 		done = np.argwhere(np.isclose(timeSpent, durations)).ravel()

# 		score += (states[done] != 1).sum() * iteration

# 		states[done] = 1

# 	return score



def rapport(optimum, solution):
	return solution/optimum


	
#Debug
import algorithm
if __name__ == "__main__":

	print("Score:", executeAlgorithm(np.array([3, 2, 2, 5, 2]), np.array([3, 3, 1, 4, 1]), algorithm.predRoundRobin) )
	#print("Score:", executeAlgorithm(np.array([3, 2, 2, 5, 2]), np.array([3, 3, 1, 4, 1]), algorithm.predRoundRobin) )


