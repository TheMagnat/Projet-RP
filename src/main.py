
import helper

import algorithm

import numpy as np




#Generation d'instances random
#n = 10
#durations    = helper.generateRandomDurations(n, maxDuration=300)
#predictions  = helper.generatePredictionsOnDurations(n, durations, variation=300)

#Exemple du sujet
durations = np.array([1, 1, 1, 2])
predictions = np.array([1, 1, 1, 1])


optScheduling = helper.genSortedScheduling(durations)
#scheduling = helper.genRandomScheduling(n)

print("Durations:", durations, "Predictions:", predictions, "Opt scheduling:", optScheduling)


randomScore = helper.executeAlgorithm(durations, predictions, algorithm.randomAlgorithm)
print("Random rez:", randomScore )


roundRobinScore = helper.executeAlgorithm(durations, predictions, algorithm.roundRobin)
print("Round-Robin rez:", roundRobinScore )

roundRobinScore = helper.executeAlgorithm(durations, predictions, algorithm.shortestPred)
print("Pred rez:", roundRobinScore )

predRoundRobinScore = helper.executeAlgorithm(durations, predictions, algorithm.predRoundRobin)
print("Pred Round-Robin rez:", predRoundRobinScore )

print("Opt rez:", helper.testScheduling(durations, optScheduling) )
