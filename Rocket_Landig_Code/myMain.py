#This is my main file for this program
# I'm imprting N wich is imprortig othe classes
# I'm also imprting all the dataset we initially divided
from myNeuralNetwork import NN
from myDataNormalization import trainingData, testingData, validationData

if __name__ == "__main__":
    # we crete our own custom lenght neural network
    myNN = NN([2, 4, 2])
    # we cll the functions of the class
    myNN.train(trainingData, 0.022553, 10000)
    myNN.testing(testingData)
    myNN.validation(validationData)
    myNN.predict(testingData)