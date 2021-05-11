
import numpy as np

LAMBDA = 0.2

"""
n: Number of task
states: array of the state of the task, 1 if finished, 0 otherwise
pred: predictions of each task

return an array of the power allocated to each task (which should sum to 1)
"""
def roundRobin(n, pred, states):

	notDone = np.argwhere(states < 1).ravel()

	size = notDone.shape[0]

	allocation = np.zeros(n, dtype=np.float64)
	allocation[notDone] = np.float64(1)/np.float64(size)

	return allocation



def randomAlgorithm(n, pred, states):

	notDone = np.argwhere(states < 1).ravel()

	allocation = np.zeros(n, dtype=np.float64)
	allocation[ notDone[np.random.randint(notDone.size)] ] = 1

	return allocation



def shortestPred(n, pred, states):

	notDone = np.argwhere(states < 1).ravel()


	argShortestNotDone = pred[notDone].argmin()

	argShortest = notDone[argShortestNotDone]

	allocation = np.zeros(n, dtype=np.float64)
	allocation[ argShortest ] = 1

	return allocation


def predRoundRobin(n, pred, states):

	#For random
	#randFloat = np.random.uniform()
	#lambdaa = randFloat

	lambdaa = LAMBDA


	roroSpeed = lambdaa
	predSpeed = 1 - lambdaa

	allocationsRoro = roundRobin(n, pred, states)

	allocationsPred = shortestPred(n, pred, states)

	allocationsShared = allocationsRoro * roroSpeed + allocationsPred * predSpeed

	return allocationsShared

	
#Debug
if __name__ == "__main__":


	predRoundRobin(5, np.array([3, 2, 1, 5, 2]), np.array([0, 0, 0, 0, 0]))

