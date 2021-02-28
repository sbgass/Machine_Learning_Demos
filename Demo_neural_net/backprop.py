#!/usr/bin/env python3
import numpy as np
from io import StringIO

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "./adult/"

#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    y = max(y,0) #treat -1 as 0 instead, because sigmoid's range is 0-1
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
        return np.asarray([ys],dtype=np.float32).T, np.asarray(xs,dtype=np.float32).reshape(len(xs),NUM_FEATURES,1) #returns a tuple, first is an array of labels, second is an array of feature vectors

def init_model(args):
    w1 = None
    w2 = None

    if args.weights_files:
        with open(args.weights_files[0], 'r') as f1:
            w1 = np.loadtxt(f1)
            f1.close()
        with open(args.weights_files[1], 'r') as f2:
            w2 = np.loadtxt(f2)
            w2 = w2.reshape(1,len(w2))
            f2.close()
    else:
        w1 = np.random.rand(args.hidden_dim, NUM_FEATURES) #bias included in NUM_FEATURES
        w2 = np.random.rand(1, args.hidden_dim + 1) #add bias column

    #At this point, w1 has shape (hidden_dim, NUM_FEATURES) and w2 has shape (1, hidden_dim + 1). In both, the last column is the bias weights.


    model = [w1.T,w2.T]
    return model

def sigmoid(x,w):
    a = 1/(1+np.exp(-np.dot(x,w)))
    return a 

def train_model(model, train_ys, train_xs, dev_ys, dev_xs, args):

    lr = args.lr 
    iterations = args.iterations
    w1, w2 = model

    for epoch in range(iterations):
        for i in range(len(train_xs)):
            ##forward pass 
            #calc hidden layer output
            z = sigmoid(train_xs[i].T,w1)  # using sigmoid as activation function 
            z = np.append(z,1) #add bias output to hidden layer 

            #calc last layer output 
            yhat = sigmoid(z.T,w2) # 1 x 6 times 6 x 1 

            ##backward pass 
            #update w2 first  
            delta2 = yhat - train_ys[i] #derivative of cross entropy wrt input to yhat * g'(a)
            #delta is a 1x1 scalar 
            w2 = np.subtract(w2, lr * (delta2 * z.reshape(len(z),1))) # 1x6 - 1x1*1x1*1x6  = 1x6 
            #then update w1
            gprime = z*(1-z) #1x5
            gprime = gprime.T[:-1].reshape(len(gprime)-1,1) #5x1 transpose and remove bias
            delta1 = delta2 * np.multiply(w2[:-1],gprime) # 1x1 times 1x5 element multiply 1x5.  
            dw1 = np.dot(delta1,train_xs[i].T) #5x1 times 1x124 = 5 x 124 
            w1 = np.subtract(w1, lr*dw1.T)
        
        #at the end of the epoch, check performance on dev data 
        if not args.nodev:
            dev_acc = test_accuracy((w1,w2),dev_ys,dev_xs)

    model = [w1,w2]
    return model

def test_accuracy(model, test_ys, test_xs):
    accuracy = 0.0
    correct = 0 
    w1, w2 = model
    for i in range(len(test_xs)):
        z= sigmoid(test_xs[i].T,w1)
        z = np.append(z,1)
        yhat = 1 if sigmoid(z.T,w2) > 0.5 else 0 #make a classification 0 or 1 for final output 

        if yhat == test_ys[i]:
            correct += 1 

    accuracy = correct/len(test_xs)
    
    return accuracy

def extract_weights(model):
    #this helper function is only used to print the weights after training
    w1 = None
    w2 = None
    w1, w2 = model 

    return w1.T, w2.T

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Neural network with one hidden layer, trainable with backpropagation.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for update in training loop.')

    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights_files', nargs=2, metavar=('W1','W2'), type=str, help='Files to read weights from (in format produced by numpy.savetxt). First is weights from input to hidden layer, second is from hidden to output.')
    weights_group.add_argument('--hidden_dim', type=int, default=5, help='Dimension of hidden layer.')

    parser.add_argument('--print_weights', action='store_true', default=False, help='If provided, print final learned weights to stdout')

    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')


    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.weights_files: iterable of str; if present, contains two fields, the first is the file to read the first layer's weights from, second is for the second weight matrix.
    args.hidden_dim: int; number of hidden layer units. If weights_files is provided, this argument should be ignored.
    args.train_file: str; file to load training data from.
    args.dev_file: str; file to load dev data from.
    args.test_file: str; file to load test data from.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)

    model = init_model(args)
    model = train_model(model, train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(model, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    if args.print_weights:
        w1, w2 = extract_weights(model)
        with StringIO() as weights_string_1:
            np.savetxt(weights_string_1,w1)
            print('Hidden layer weights: {}'.format(weights_string_1.getvalue()))
        with StringIO() as weights_string_2:
            np.savetxt(weights_string_2,w2)
            print('Output layer weights: {}'.format(weights_string_2.getvalue()))

if __name__ == '__main__':
    main()
