#!/usr/bin/python3
import numpy as np

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "./adult/"

#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray(ys), np.asarray(xs) #returns a tuple, first is an array of labels, second is an array of feature vectors

def perceptron(train_ys, train_xs, dev_ys, dev_xs, args):
    weights = np.zeros(NUM_FEATURES)
    #Returns a numpy array of trained weights
    
    lr = args.lr #float
    num_iter = args.iterations #intiger
    train_accuracy = []
    dev_accuracy = []
    #dev_ys or dev_xs == None then no development data
    for epoch in range(num_iter):
        #making predictions
        yhat = np.zeros(len(train_ys))
        dev_correct = 0
        train_correct = 0
        for i in range(len(train_xs)): #for each data point
            ti = np.dot(weights, train_xs[i]) #make prediction
            if ti > 0:
                yhat[i] = 1
            elif ti < 0:
                yhat[i] = -1
            train_correct += 1 if yhat[i] == train_ys[i] else 0
            weights += lr*train_ys[i]*train_xs[i] if yhat[i] != train_ys[i] else 0
        #record accuracy
        train_accuracy.append(train_correct / len(train_ys))
        
        
        
        #Test accuracy of dev data
        if not args.nodev:
            for i in range(len(dev_xs)):
                tdi = np.dot(weights,dev_xs[i])
                if tdi > 0:
                    tdi = 1
                elif tdi < 0:
                    tdi = -1
                else:
                    tdi = 0
                dev_correct += 1 if dev_ys[i] == tdi else 0

            dev_accuracy.append(dev_correct / len(dev_ys))
        
        
    #
    
    return weights

def test_accuracy(weights, test_ys, test_xs):
    accuracy = 0.0
    #Returns a float of the accuracy of the model on the test data 
    
    #making predictions
    yhat = np.zeros(len(test_ys))
    for i in range(len(test_xs)):
        ti = np.dot(weights, test_xs[i])
        if ti > 0:
            yhat[i] = 1
        elif ti < 0:
            yhat[i] = -1
        
        
    #calculate accuracy
    correct = 0
    for i in range(len(yhat)):
        if test_ys[i] == yhat[i]:
            correct += 1
    accuracy = correct/len(yhat)
    
    
    return accuracy

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate to use for update in training loop.')
    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')
    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.train_file: str; file name for training data.
    args.dev_file: str; file name for development data.
    args.test_file: str; file name for test data.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)
    weights = perceptron(train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(weights, test_ys, test_xs)
    print('Test accuracy: {}'.format(round(accuracy,8)))
    print('Feature weights (bias last): {}'.format(' '.join(map(str,weights))))

if __name__ == '__main__':
    main()
