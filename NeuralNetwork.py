from Perceptron import Perceptron
from Layer import Layer

class NeuralNetwork:
    layerList: list[Layer]

    def __init__(self, inputAmount: int, perceptronsPerLayerList: list[int]) -> None:
        self.layerList: list[Layer] = []

        for perceptronAmount in perceptronsPerLayerList:
            layer = Layer(inputAmount, perceptronAmount)
            inputAmount = perceptronAmount
            self.layerList.append(layer)

        
    def classify(self, processorYear: float, processorTDP: float) -> list[float]:
        inputList = [processorYear, processorTDP]
        
        for idx, layer in enumerate(self.layerList):
            if(idx + 1 < len(self.layerList)):
                inputList = layer.calculateOutputList(inputList, True)
            else:
                inputList = layer.calculateOutputList(inputList, False)
            
        return inputList[0]
    

    def calculateError(self, dataset: list[list[float]]):

        error = 0.0
        for inputList in dataset:   
            result = self.classify(inputList[0], inputList[1])
            expectedResult = inputList[2]

            error += ((result - expectedResult) **  2) / len(dataset)

        return error
    
    def trainNetwork(self, dataset: list[list[float]], learnRate: float = 0.00000000005):
        
        adjustmentRate = 0.0001
        originalError = self.calculateError(dataset)

        for layer in self.layerList:
            for perceptron in layer.perceptronList:
                for i in range(len(perceptron.weightList)):
                    perceptron.weightList[i] += adjustmentRate
                    newError = self.calculateError(dataset)
                    perceptron.weightList[i] -= adjustmentRate

                    difference = min(1, (newError - originalError) / adjustmentRate)
                    difference = max(-1, difference)
                    perceptron.weightList[i] -= difference * learnRate

                perceptron.bias += adjustmentRate
                newError = self.calculateError(dataset)
                perceptron.bias -+ adjustmentRate

                difference = min(1, (newError - originalError) / adjustmentRate)
                difference = max(-1, difference)

                perceptron.bias -= difference * learnRate * 10
    
    