from math import sqrt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import random
import math


# region SearchAlgorithms


class Node:
    id = None
    up = None
    down = None
    left = None
    right = None
    previousNode = None

    def __init__(self, value):
        self.value = value


class SearchAlgorithms:
    """ * DON'T change Class, Function or Parameters Names and Order
        * You can add ANY extra functions,
          classes you need as long as the main
          structure is left as is """
    path = []  # Represents the correct path from start node to the goal node.
    fullPath = []  # Represents all visited nodes from the start node to the goal node.
    maze = []  # The 2D array Map
    rows = 0  # Number of rows
    columns = 0  # Number of columns

    def __init__(self, mazeStr):
        """ mazeStr contains the full board
         The board is read row wise,
        the nodes are numbered 0-based starting
        the leftmost node"""
        self.maze = [j.split(',') for j in mazeStr.split(' ')]
        # get number of columns
        for i in mazeStr:
            if i == ',':
                self.columns += 1
            if i == ' ':
                break
        # get number of rows
        for i in mazeStr:
            if i == ' ':
                self.rows += 1
        pass

    def BFS(self):
        """Implement Here"""
        queue = []
        visited = []
        x = 0
        y = 0
        for i in range(self.rows):
            for j in range(self.columns):
                if self.maze[i][j] == 'S':
                    x = i
                    y = j
        currentNode = self.maze[x][y]
        n = Node(currentNode)
        n.id = (x, y)
        queue.append(n)
        while currentNode != 'E':
            n = queue.pop(0)
            x = n.id[0]
            y = n.id[1]
            self.fullPath.append((x * (self.columns + 1)) + y)
            currentNode = self.maze[x][y]
            if x + 1 <= self.rows:
                if (x + 1, y) not in visited:
                    n.up = (x + 1, y)
                    if self.maze[x + 1][y] != '#':
                        n1 = Node(self.maze[x + 1][y])
                        n1.id = (x + 1, y)
                        n1.previousNode = n
                        queue.append(n1)
                        visited.append((x + 1, y))
            if x - 1 >= 0:
                if (x - 1, y) not in visited:
                    n.down = (x - 1, y)
                    if self.maze[x - 1][y] != '#':
                        n1 = Node(self.maze[x - 1][y])
                        n1.id = (x - 1, y)
                        queue.append(n1)
                        n1.previousNode = n
                        visited.append((x - 1, y))

            if y - 1 >= 0:
                if (x, y - 1) not in visited:
                    n.left = (x, y - 1)
                    if self.maze[x][y - 1] != '#':
                        n1 = Node(self.maze[x][y - 1])
                        n1.id = (x, y - 1)
                        queue.append(n1)
                        n1.previousNode = n
                        visited.append((x, y - 1))
            if y + 1 <= self.columns:
                if (x, y + 1) not in visited:
                    n.right = (x, y + 1)
                    if self.maze[x][y + 1] != '#':
                        n1 = Node(self.maze[x][y + 1])
                        n1.id = (x, y + 1)
                        queue.append(n1)
                        n1.previousNode = n
                        visited.append((x, y + 1))
            visited.append((x, y))
        self.path.append((n.id[0] * (self.columns + 1)) + n.id[1])
        while n.value != 'S':
            n = n.previousNode
            self.path.append((n.id[0] * (self.columns + 1)) + n.id[1])
        self.path.reverse()
        return self.fullPath, self.path


# endregion

# region NeuralNetwork
class NeuralNetwork():

    def __init__(self, learning_rate, threshold):
        self.learning_rate = learning_rate
        self.threshold = threshold
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((2, 1)) - 1

    def step(self, x):
        if x > float(self.threshold):
            return 1
        else:
            return 0
        pass

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            self.synaptic_weights += np.dot(training_inputs.T, error * self.learning_rate)
        pass

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.step(np.sum(np.dot(inputs, self.synaptic_weights)))
        return output
        pass


# endregion


# region ID3
class item:
    def __init__(self, age, prescription, astigmatic, tearRate, diabetic, needLense):
        self.age = age
        self.prescription = prescription
        self.astigmatic = astigmatic
        self.tearRate = tearRate
        self.diabetic = diabetic
        self.needLense = needLense
    def getDataset():
        data = []
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
        data.append(item(0, 0, 0, 0, 1, labels[0]))
        data.append(item(0, 0, 0, 1, 1, labels[1]))
        data.append(item(0, 0, 1, 0, 1, labels[2]))
        data.append(item(0, 0, 1, 1, 1, labels[3]))
        data.append(item(0, 1, 0, 0, 1, labels[4]))
        data.append(item(0, 1, 0, 1, 1, labels[5]))
        data.append(item(0, 1, 1, 0, 1, labels[6]))
        data.append(item(0, 1, 1, 1, 1, labels[7]))
        data.append(item(1, 0, 0, 0, 1, labels[8]))
        data.append(item(1, 0, 0, 1, 1, labels[9]))
        data.append(item(1, 0, 1, 0, 1, labels[10]))
        data.append(item(1, 0, 1, 1, 0, labels[11]))
        data.append(item(1, 1, 0, 0, 0, labels[12]))
        data.append(item(1, 1, 0, 1, 0, labels[13]))
        data.append(item(1, 1, 1, 0, 0, labels[14]))
        data.append(item(1, 1, 1, 1, 0, labels[15]))
        data.append(item(1, 0, 0, 0, 0, labels[16]))
        data.append(item(1, 0, 0, 1, 0, labels[17]))
        data.append(item(1, 0, 1, 0, 0, labels[18]))
        data.append(item(1, 0, 1, 1, 0, labels[19]))
        data.append(item(1, 1, 0, 0, 0, labels[20]))
        return data

class Feature:
    def __init__(self, name):
        self.name = name
        self.visited = -1
        self.infoGain = -1

class ID3:
    dataset = item.getDataset()
    def __init__(self, features):
        self.features = features
        self.buildtree('Root')
    def entropy(self,list):
        numberOFZeros = 0
        numberOFOnes = 0
        for i in range(0, len(list)):
            if list[i] == 0:
                numberOFZeros += 1
            elif list[i] == 1:
                numberOFOnes += 1
        length = numberOFOnes + numberOFZeros
        if length == 0:
            return 0
        elif numberOFZeros == 0 and numberOFOnes == 0:
            return 0
        elif numberOFZeros == 0 and numberOFOnes != 0:
            return -((numberOFOnes / length) * math.log2(numberOFOnes / length))
        elif numberOFZeros != 0 and numberOFOnes == 0:
            return -((numberOFZeros / length) * math.log2(numberOFZeros / length))
        else:
            return -(((numberOFZeros / length) * math.log2(numberOFZeros / length)) + ((numberOFOnes / length) * math.log2(numberOFOnes / length)))
    def Get_Max_Gain(self):
        Max_Gain = -99999
        age = []
        prescription = []
        astigmatic = []
        tearRate = []
        diabetic = []
        needLenses = []
        maxColumn = 0
        for i in range(0,len(self.dataset)):
            age.append(self.dataset[i].age)
            prescription.append(self.dataset[i].prescription)
            astigmatic.append(self.dataset[i].astigmatic)
            tearRate.append(self.dataset[i].tearRate)
            diabetic.append(self.dataset[i].diabetic)
            needLenses.append(self.dataset[i].needLense)
        listOFColumns = [age,prescription,astigmatic,tearRate,diabetic,needLenses]
        for i in range(0,5):
            if self.features[i].visited == -1:
                numberOfZeros = 0
                numberOfOnes = 0
                listOFZeros = []
                listOFOnes = []
                length = len(listOFColumns[i])
                for j in range(0, length):
                    if listOFColumns[i][j] == 0:
                        numberOfZeros += 1
                        listOFZeros.append(listOFColumns[5][j])
                    elif listOFColumns[i][j] == 1:
                        numberOfOnes += 1
                        listOFOnes.append(listOFColumns[5][j])
                gain = self.entropy(listOFColumns[5]) - (((numberOfZeros / length) * self.entropy(listOFZeros)) +
                                                         ((numberOfOnes / length) * self.entropy(listOFOnes)))
                if gain > Max_Gain:
                    Max_Gain = gain
                    maxColumn = i
        self.features[maxColumn].visited = 1
        return maxColumn

    startNode = None
    currentNode = None
    def buildtree(self,postion):

        maxGainColumn = self.Get_Max_Gain()
        print(maxGainColumn)
        node = Node(maxGainColumn)
        node.id = maxGainColumn
        if postion == 'Root':
            self.startNode = node
        elif postion == 'Left':
            self.currentNode.left = node
        elif postion == 'Right':
            self.currentNode.right = node

        self.currentNode = node
        datasetZeros = []
        datasetOnes = []
        tmpResultZeros = []
        tmpResultones = []

        if maxGainColumn == 0:
            for i in range(0, len(self.dataset)):
                if self.dataset[i].age == 0:
                    datasetZeros.append(self.dataset[i])
                    tmpResultZeros.append(self.dataset[i].needLense)
                elif self.dataset[i].age == 1:
                    datasetOnes.append(self.dataset[i])
                    tmpResultones.append(self.dataset[i].needLense)
        elif maxGainColumn == 1:
            for i in range(0, len(self.dataset)):
                if self.dataset[i].prescription == 0:
                    datasetZeros.append(self.dataset[i])
                    tmpResultZeros.append(self.dataset[i].needLense)
                elif self.dataset[i].prescription == 1:
                    datasetOnes.append(self.dataset[i])
                    tmpResultones.append(self.dataset[i].needLense)
        elif maxGainColumn == 2:
            for i in range(0, len(self.dataset)):
                if self.dataset[i].astigmatic == 0:
                    datasetZeros.append(self.dataset[i])
                    tmpResultZeros.append(self.dataset[i].needLense)
                elif self.dataset[i].astigmatic == 1:
                    datasetOnes.append(self.dataset[i])
                    tmpResultones.append(self.dataset[i].needLense)
        elif maxGainColumn == 3:
            for i in range(0, len(self.dataset)):
                if self.dataset[i].tearRate == 0:
                    datasetZeros.append(self.dataset[i])
                    tmpResultZeros.append(self.dataset[i].needLense)
                elif self.dataset[i].tearRate == 1:
                    datasetOnes.append(self.dataset[i])
                    tmpResultones.append(self.dataset[i].needLense)
        elif maxGainColumn == 4:
            for i in range(0, len(self.dataset)):
                if self.dataset[i].diabetic == 0:
                    datasetZeros.append(self.dataset[i])
                    tmpResultZeros.append(self.dataset[i].needLense)
                elif self.dataset[i].diabetic == 1:
                    datasetOnes.append(self.dataset[i])
                    tmpResultones.append(self.dataset[i].needLense)

        self.dataset.clear()
        tmpResult = np.unique(tmpResultZeros)
        if len(tmpResult) >1:
            self.dataset = datasetZeros.copy()
            self.buildtree('Left')
        else:
            self.currentNode.left = tmpResult[0]
        tmpResult = np.unique(tmpResultones)
        if len(tmpResult) > 1:
            for i in range(0,len(datasetOnes)):
                self.dataset = datasetOnes.copy()
            self.buildtree('Right')
        else:
            self.currentNode.right = tmpResult[0]

    def classify(self, input):
        node = self.startNode
        print(node.id)
        print(node.left)
        while True:
            if node.id == 0:
                if input[0] == 1:
                    if node.right == 0 or node.right == 1:
                        return node.right
                    else:
                        node = node.right
                elif input[0] == 0:
                    if node.left == 0 or node.left == 1:
                        return node.left
                    else:
                        node = node.left
            elif node.id == 1:
                if input[1] == 1:
                    if node.right == 0 or node.right == 1:
                        return node.right
                    else:
                        node = node.right
                elif input[1] == 0:
                    if node.left == 0 or node.left == 1:
                        return node.left
                    else:
                        node = node.left
            elif node.id == 2:
                if input[2] == 1:
                    if node.right == 0 or node.right == 1:
                        return node.right
                    else:
                        node = node.right
                elif input[2] == 0:
                    if node.left == 0 or node.left == 1:
                        return node.left
                    else:
                        node = node.left
            elif node.id == 3:
                if input[3] == 1:
                    if node.right == 0 or node.right == 1:
                        return node.right
                    else:
                        node = node.right
                elif input[3] == 0:
                    if node.left == 0 or node.left == 1:
                        return node.left
                    else:
                        node = node.left
            elif node.id == 4:
                if input[4] == 1:
                    if node.right == 0 or node.right == 1:
                        return node.right
                    else:
                        node = node.right
                elif input[4] == 0:
                    if node.left == 0 or node.left == 1:
                        return node.left
                    else:
                        node = node.left

# endregion

#################################### Algorithms Main Functions #####################################
# region Search_Algorithms_Main_Fn

def SearchAlgorithm_Main():
    searchAlgo = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    fullPath, path = searchAlgo.BFS()
    print('**BFS**\n Full Path is: ' + str(fullPath) + "\n Path: " + str(path))


# endregion

# region Neural_Network_Main_Fn
def NN_Main():
    learning_rate = 0.1
    threshold = -0.2
    neural_network = NeuralNetwork(learning_rate, threshold)

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1]])

    training_outputs = np.array([[0, 0, 0, 1]]).T

    neural_network.train(training_inputs, training_outputs, 100)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    inputTestCase = [1, 1]

    print("Considering New Situation: ", inputTestCase[0], inputTestCase[1], end=" ")
    print("New Output data: ", end=" ")
    print(neural_network.think(np.array(inputTestCase)))
    print("Wow, we did it!")


# endregion
# region ID3_Main_Fn
def ID3_Main():
    dataset = item.getDataset()
    features = [Feature('age'), Feature('prescription'), Feature('astigmatic'), Feature('tearRate'),
                Feature('diabetic')]
    id3 = ID3(features)
    cls = id3.classify([0, 0, 1, 1, 1])
    print('testcase 1: ', cls)
    cls = id3.classify([1, 1, 0, 0, 0])
    print('testcase 2: ', cls)
    cls = id3.classify([1, 1, 1, 0, 0])
    print('testcase 3: ', cls)
    cls = id3.classify([1, 1, 0, 1, 0])
    print('testcase 4: ', cls)


# endregion

######################## MAIN ###########################33
if __name__ == '__main__':
    SearchAlgorithm_Main()
    NN_Main()
    ID3_Main()
