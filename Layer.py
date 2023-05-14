from Perceptron import Perceptron
import random

class Layer:

    def __init__(self, inputAmount: int, outputAmount: int) -> None:
        self.perceptronList: list[Perceptron] = []

        random.seed(6000)
        for i in range(outputAmount):
            bias = (random.random() - 0.5) * 200
            weightList: list[float] = []
            for j in range(inputAmount):
                weightList.append((random.random() - 0.5) * 0.02)
            perceptron = Perceptron(weightList, bias)
            self.perceptronList.append(perceptron)

        pass


    def calculateOutputList(self, inputList: list[float], activated: bool = True) -> list[float]:
        outputList: list[float] = []

        for perceptron in self.perceptronList:
            perceptronOutput = perceptron.calculateOutput(inputList, activated)
            outputList.append(perceptronOutput)

        return outputList