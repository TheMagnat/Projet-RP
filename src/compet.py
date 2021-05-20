
import helper

import algorithm

import numpy as np

#Generation d'instances random
n = 50
#error = 0.1


testRange = 1000

results = np.empty((testRange, 6), dtype=np.float64)

errors = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1]

for err in errors:

	for i in range(testRange):

		#Generate testing samples
		durations   = helper.generateRandomParetoDurations(n)
		predictions = helper.generateNormalPredictionsOnDurations(n, durations, sigma=err)
		arrive 		= helper.generateRandomArrives(n, exposant=0.92)

		#Part 1
		optScheduling = helper.genSortedScheduling(durations)
		opt = helper.testScheduling(durations, optScheduling)

		roundRobinScore = helper.executeAlgorithm(durations, predictions, algorithm.roundRobin)
		results[i, 0] = helper.rapport(optimum=opt, solution=roundRobinScore)

		predScore = helper.executeAlgorithm(durations, predictions, algorithm.shortestPred)
		results[i, 1] = helper.rapport(optimum=opt, solution=predScore)

		predRoundRobinScore = helper.executeAlgorithm(durations, predictions, algorithm.predRoundRobin)
		results[i, 2] = helper.rapport(optimum=opt, solution=predRoundRobinScore)


		#Part 2 - With arrive
		opt2 = helper.sortedDurationWithArrive(durations, arrive)

		roundRobinScore = helper.executeAlgorithm(durations, predictions, algorithm.roundRobin, arrive=arrive)
		results[i, 3] = helper.rapport(optimum=opt2, solution=roundRobinScore)

		predScore = helper.executeAlgorithm(durations, predictions, algorithm.shortestPred, arrive=arrive)
		results[i, 4] = helper.rapport(optimum=opt2, solution=predScore)

		predRoundRobinScore = helper.executeAlgorithm(durations, predictions, algorithm.predRoundRobin, arrive=arrive)
		results[i, 5] = helper.rapport(optimum=opt2, solution=predRoundRobinScore)


	print(f"Mean competitivity of each algorithm with error = {err}:")

	labels = ["Round-robin", "Predictions", "Pred Round-robin", "Round-robin with arrive", "Predictions with arrive", "Pred Round-robin with arrive"]
	for label, res in zip(labels, results.mean(axis=0)):
		print(label, res)

	print()

