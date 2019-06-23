import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt

def netFunction(input, bias, weight): # Function to calculate net
	return bias + np.matmul(input,weight)

def unitStepFunction(net): # Function to decide output (-1 or 1)
	return np.where(net >= 0.0, 1, -1)

def boundaryFunction(x, bias, weight): # Function to plot decision boundary
	return -1*(bias/weight[1]) + np.dot(x,-1*(weight[0]/weight[1]))

def costFunction(X, Y, bias, weight): # Cost Function
	sum = 0
	for x, y in zip (X,Y):
		net = netFunction(x, bias, weight)
		error = y - net
		sum = sum + error**2
	return 0.5 * sum

def splitDataIntoTwoClases(X,Y): # split input data by target
	x1_class1 = X['x1'].loc[Y==1]
	x1_class2 = X['x1'].loc[Y==-1]
	x2_class1 = X['x2'].loc[Y==1]
	x2_class2 = X['x2'].loc[Y==-1]
	return x1_class1, x1_class2, x2_class1, x2_class2

def plotData(X,Y,figureIndex): # plot data of coordinate X with label Y
	# dividing data into two classes
	x1_class1, x1_class2, x2_class1, x2_class2 = splitDataIntoTwoClases(X,Y)

	# scatter plot data
	plt.figure(figureIndex)
	plt.scatter(x1_class1,x2_class1,alpha=0.5,c='#FFA500')
	plt.scatter(x1_class2,x2_class2,alpha=0.5,c='cyan')
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.legend(['Class 1','Class 2'])

def plotDecisionBoundary(X,Y,bias,weight,figureIndex): # plot Decision Boundary
	# dividing data into two classes
	x1_class1, x1_class2, x2_class1, x2_class2 = splitDataIntoTwoClases(X,Y)

	maxValue = X['x1'].max()
	minValue = X['x2'].min()
	x1_line = np.array([minValue,maxValue])
	x2_line = boundaryFunction(x1_line,bias,weight)

	# scatter plot data
	plt.figure(figureIndex)
	plt.scatter(x1_class1,x2_class1,alpha=0.5,c='#FFA500')
	plt.scatter(x1_class2,x2_class2,alpha=0.5,c='cyan')
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.legend(['Class 1','Class 2'])
	plt.plot(x1_line,x2_line)

def train(X, Y, bias, weight, rate, epochs): # Training process by gradient descent
	for iterEpoch in range(epochs):
		for input,target in zip (X,Y):
			net = netFunction(input,bias,weight)

			error = target - net
			bias = bias + rate * error
			
			delta = np.dot(rate * error,input)
			weight = weight + delta
		print (costFunction(X, Y, bias, weight))
	return bias, weight

def predict(X,bias,weight): # Predict label given by X
	output = []
	for x in X:
		net = netFunction(x,bias,weight)
		o = unitStepFunction(net)
		output.append(o)
	return output

def accuracy(output,Y): # Calculating the accuracy
	margin = Y - output
	success = len([i for i in margin if i == 0])
	return success / len(output)

def main():
    print("Adaptive Linear Networks (ADALINE) Gradient Descent")
    
    # read data file
    filePath = '../data/data.csv' # path location of 'data.csv'
    rawData = pd.read_csv(filePath) # read csv data based on filePath
    
    # split raw data into input data (X) and target (Y)
    X = rawData[['x1','x2']]
    Y = rawData['y']

    # # split dataset into training dataset and testing dataset
    # msk = np.random.rand(len(X)) < 0.8 #generate random array numbers
    # # spliting for training data set
    # X_train = X[msk]
    # Y_train = Y[msk]
    # # spliting for testing data set
    # X_test = X[~msk]
    # Y_test = Y[~msk]

    dataSize = X.shape
    fold = 0.8
    splitIndex = mt.floor(dataSize[0]*fold)
    X_train = X.iloc[0:splitIndex,:]
    Y_train = Y[0:splitIndex]
    X_test = X.iloc[splitIndex+1:dataSize[0],:]
    Y_test = Y[splitIndex+1:dataSize[0]]

    # training the model
    # initilize parameter
    learningRate = 0.0005
    epoch = 100
    bias = np.random.random_sample()
    weight = np.random.random(2)

    # train the model
    bias, weight = train(X_train.values, Y_train.values, bias, weight, learningRate, epoch)
    print(bias)
    print(weight)

    # testing
    output = predict(X_test.values,bias,weight)

    # accuracy
    acc = accuracy(output, Y_test.values)
    print ("Accuracy = %.3f" % acc)

    # plot data
    plotData(X,Y,1)

    # plot decision boundary line
    plotDecisionBoundary(X,Y,bias,weight,2)

    # show the figures
    plt.show()


if __name__ == "__main__":
    main()
