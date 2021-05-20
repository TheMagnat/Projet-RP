
import helper

import algorithm

import numpy as np

#Generation d'instances random
n = 7
error = 0.1

#For integer durations
#durations    = helper.generateRandomDurations(n, maxDuration=300)
durations    = helper.generateRandomParetoDurations(n)

#For integer predictions
#predictions  = helper.generatePredictionsOnDurations(n, durations, variation=300)
predictions  = helper.generateNormalPredictionsOnDurations(n, durations, sigma=error)

#For integer arrives
#arrive = helper.generateRandomArrivesInteger(n, variation=300)
arrive = helper.generateRandomArrives(n, exposant=0.92)

durations = np.array([1, 1, 1, 2], dtype=np.float64)
predictions = np.array([1, 2, 1, 1], dtype=np.float64)
arrive = np.array([0, 3, 5, 0], dtype=np.float64)

print("durations:", durations)
print(f"predictions (error: {error}):", predictions)

#Part 1
optScheduling = helper.genSortedScheduling(durations)
opt = helper.testScheduling(durations, optScheduling)
print("Opt rez:",  opt)

roundRobinScore = helper.executeAlgorithm(durations, predictions, algorithm.roundRobin)
print("Round-Robin rez:", roundRobinScore, helper.rapport(optimum=opt, solution=roundRobinScore))

predScore = helper.executeAlgorithm(durations, predictions, algorithm.shortestPred)
print("Pred rez:", predScore, helper.rapport(optimum=opt, solution=predScore))

predRoundRobinScore = helper.executeAlgorithm(durations, predictions, algorithm.predRoundRobin)
print("Pred Round-Robin rez:", predRoundRobinScore, helper.rapport(optimum=opt, solution=predRoundRobinScore))




#Part 2 - With arrive
print("\nNow with arrive time:", arrive)

opt2 = helper.sortedDurationWithArrive(durations, arrive)
print("Opt rez:", opt2)

roundRobinScore = helper.executeAlgorithm(durations, predictions, algorithm.roundRobin, arrive=arrive)
print("B - Round-Robin rez:", roundRobinScore, helper.rapport(optimum=opt2, solution=roundRobinScore))

predScore = helper.executeAlgorithm(durations, predictions, algorithm.shortestPred, arrive=arrive)
print("B - Pred rez:", predScore, helper.rapport(optimum=opt2, solution=predScore))

predRoundRobinScore = helper.executeAlgorithm(durations, predictions, algorithm.predRoundRobin, arrive=arrive)
print("B - Pred Round-Robin rez:", predRoundRobinScore, "Compet:", helper.rapport(optimum=opt2, solution=predRoundRobinScore))

