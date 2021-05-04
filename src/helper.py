
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
	while states.sum() < n:

		iteration += 1

		allocation = algorithm(n, predictions, states.copy())

		if allocation.sum() > 1:
			print("Allocation > 1, cheating")

		timeSpent += allocation

		done = np.argwhere(np.isclose(timeSpent, durations)).ravel()

		score += (states[done] != 1).sum() * iteration

		states[done] = 1

	return score


def rapport(optimum, solution):
	return solution/optimum



