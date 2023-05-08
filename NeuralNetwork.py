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
        
        for layer in self.layerList:
            inputList = layer.calculateOutputList(inputList, True)
            
        return inputList[0]
    

    def calculateError(self, dataset: list[list[float]]):

        error = 0.0
        for inputList in dataset:   
            result = self.classify(inputList[0], inputList[1])
            expectedResult = inputList[2]

            error += (result - expectedResult) **  2

        return error
    
    def trainNetwork(self, dataset: list[list[float]], learnRate: float = 0.00000000005):
        
        adjustmentRate = 0.001
        originalError = self.calculateError(dataset)

        for layer in self.layerList:
            for perceptron in layer.perceptronList:
                for i in range(len(perceptron.weightList)):
                    # print("orrError: ", originalError)
                    perceptron.weightList[i] += adjustmentRate
                    newError = self.calculateError(dataset)
                    perceptron.weightList[i] -= adjustmentRate
                    # print("newError: ", newError)

                    difference = (newError - originalError) / adjustmentRate 
                    # print("difference: ", difference)
                    # print("original Weight: ", perceptron.weightList[i])
                    perceptron.weightList[i] -= difference * learnRate
                    # print("new Weight: ", perceptron.weightList[i])

                perceptron.bias += adjustmentRate * 100
                newError = self.calculateError(dataset)
                perceptron.bias -+ adjustmentRate * 100

                difference = (newError - originalError) / adjustmentRate

                perceptron.bias -= difference * learnRate 
    
    