
import numpy as np

"""
n: Number of task
states: array of the state of the task, 1 if finished, 0 otherwise

return an array of the power allocated to each task (which should sum to 1)
"""
def roundRobin(n, states):

	notDone = np.argwhere(states < 1).ravel()

	size = notDone.shape[0]

	allocation = np.zeros(n, dtype=np.float64)
	allocation[notDone] = np.float64(1)/np.float64(size)

	return allocation



def randomAlgorithm(n, states):

	notDone = np.argwhere(states < 1).ravel()

	allocation = np.zeros(n, dtype=np.float64)
	allocation[ notDone[np.random.randint(notDone.size)] ] = 1

	return allocation