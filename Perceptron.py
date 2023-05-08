import activations as activations

class Perceptron:

    def __init__(self, weightList: list[float], bias:float = 1.0) -> None:
        self.weightList = weightList
        self.bias = bias

    def calculateOutput(self, inputList, activated: bool = True) -> float:
        weightedInput: float = self.bias
        for i in range(len(inputList)):
            weightedInput += inputList[i] * self.weightList[i]
        
        return activations.reLu(weightedInput)
    
    def adjustweights(self, weightList: list[float]):
        self.weightList = weightList
        pass