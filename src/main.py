
import helper

import algorithm




n = 10

durations    = helper.generateRandomDurations(n)
predictions  = helper.generatePredictionsOnDurations(n, durations)

#Exemple du sujet
#durations = np.array([1, 1, 1, 2])


optScheduling = helper.genSortedScheduling(durations)
#scheduling = helper.genRandomScheduling(n)

print("Durations:", durations, "Opt scheduling:", optScheduling)



randomScore = helper.executeAlgorithm(durations, predictions, algorithm.randomAlgorithm)
print("Random rez:", randomScore )


roundRobinScore = helper.executeAlgorithm(durations, predictions, algorithm.roundRobin)
print("Round-Robin rez:", roundRobinScore )

roundRobinScore = helper.executeAlgorithm(durations, predictions, algorithm.shortestPred)
print("Pred rez:", roundRobinScore )

print("Opt rez:", helper.testScheduling(durations, optScheduling) )
