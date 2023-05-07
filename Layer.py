from Perceptron import Perceptron

class Layer:

    def __init__(self, inputAmount: int, outputAmount: int) -> None:
        self.perceptronList: list[Perceptron] = []

        defaultweight: float = 0.1
        defaultBias: float = 1.0

        for i in range(outputAmount):
            weightList = [defaultweight] * inputAmount
            perceptron = Perceptron(weightList, defaultBias)
            self.perceptronList.append(perceptron)

        pass


    def calculateOutputList(self, inputList: list[float], activated: bool = True) -> list[float]:
        outputList: list[float] = []

        for perceptron in self.perceptronList:
            perceptronOutput = perceptron.calculateOutput(inputList, activated)
            outputList.append(perceptronOutput)

        return outputList