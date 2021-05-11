
import numpy as np

"""
arg:
	n: Number of tasks
	maxDuration: The duration are in [1, maxDuration[

return duration array
"""
def generateRandomDurations(n, maxDuration=3):

	return np.random.randint(1, maxDuration, size=n)


def generatePredictionsOnDurations(n, durations, variation=2):

	ret = durations + np.random.randint(-variation, variation+1, size=n)
	
	return np.where(ret < 1, 1, ret)



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



"""
Execute an algorithm on an instance "durations".

The algorithm take the number of task, the predictions and a state array
(which tell if a task if finished or not)
and should return an array which contain
the fraction of the computer allocated to the task at index i.
It sum should be equal to 1.

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


	print("Score:",executeAlgorithm2(np.array([3, 2, 2, 5, 2]), np.array([3, 3, 1, 4, 1]), algorithm.predRoundRobin) )


