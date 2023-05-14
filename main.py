import pickle as pickle

from NeuralNetwork import NeuralNetwork


dataset = [[2005, 90, 3900], [2000, 80, 2800], [2008, 70, 3500], [2012, 65, 3600], [2013, 80, 4200], [2006, 90, 4000], [2002, 80, 3000], [2009, 70, 3700], [2013, 65, 3800], [2014, 80, 4300]]

neuralNet = NeuralNetwork(2, [3, 1])

print("Previsão inicial para ano 2000 e TDP 80W: ", neuralNet.classify(2000, 80), " Mhz")
print("Previsão inicial para ano 2005 e TDP 90W: ", neuralNet.classify(2005, 90), " Mhz")

# for i in range(10000000):
#     if(i % 2 == 0):
#         learnDataset = dataset[:5]
#     else:
#         learnDataset = dataset[5:]
#     neuralNet.trainNetwork(learnDataset, learnRate = 0.0003)



# with open("melhorAteAgora.rn", "wb") as saveFile:
#     pickle.dump(neuralNet, saveFile, pickle.HIGHEST_PROTOCOL)

with open("melhorAteAgoraFinal.rn", "rb") as saveFile:
    neuralNet = pickle.load(saveFile)


print("Resultado previsto para ano 2000 e TDP 80W: ", neuralNet.classify(2000, 80), " Mhz")
print("Resultado previsto para ano 2005 e TDP 90W: ", neuralNet.classify(2005, 90), " Mhz")