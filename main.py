import matplotlib.pyplot as plt
import pickle as pickle

from Perceptron import Perceptron
from Layer import Layer
from NeuralNetwork import NeuralNetwork

inputYears = [2000, 2005, 2008, 2012, 2013]
inputTDP = [80, 90, 70, 65, 80]
outputGhz = [3.0, 4.0, 3.5, 3.6, 4.2]

# plt.plot()
# plt.ylim(40, 100)
# plt.xlim(1998, 2015)
# plt.scatter(inputYears, inputTDP)
# for i in range(len(outputGhz)):
#     plt.annotate(outputGhz[i], (inputYears[i] + 0.2, inputTDP[i] + 1))
# plt.show()

dataset = [[2005, 90, 4000.0], [2000, 80, 3000.0], [2008, 70, 3500.0], [2012, 65, 3600.0], [2013, 80, 4200.0]]

neuralNet = NeuralNetwork(2, [3, 1])

# print(neuralNet.calculateError(dataset))
# print(neuralNet.classify(2005, 90), " Mhz")

for i in range(10000):
    neuralNet.trainNetwork(dataset, learnRate = 0.00000000001)


# with open("redeNeural.rn", "wb") as saveFile:
#     pickle.dump(neuralNet, saveFile, pickle.HIGHEST_PROTOCOL)

# with open("melhorAteAgora.rn", "rb") as saveFile:
#     neuralNet = pickle.load(saveFile)

# print(neuralNet.calculateError(dataset))

# print(neuralNet.layerList[0].perceptronList[0].weightList, neuralNet.layerList[0].perceptronList[0].bias)
# print(neuralNet.layerList[0].perceptronList[1].weightList, neuralNet.layerList[0].perceptronList[1].bias)
# print(neuralNet.layerList[0].perceptronList[2].weightList, neuralNet.layerList[0].perceptronList[2].bias)
# print(neuralNet.layerList[1].perceptronList[0].weightList, neuralNet.layerList[1].perceptronList[0].bias)

print(neuralNet.classify(2005, 90), " Mhz")