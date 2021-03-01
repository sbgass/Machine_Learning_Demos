import numpy as np 
import time, sys 

filename = sys.argv[1]
start = time.perf_counter()

def sigma(w,x):
    import numpy as np
    #this is the activation function
    return 1 / ( 1 + np.exp(-np.dot(x,w)))

#read data file 
x= [] #input
y = [] #class label for each input 
with open(filename, 'r') as file:
    for line in file.readlines():
        line=line.split(',')
        yi = float(line[-1]) #removing the class label in the input 
        line = [float(x) for x in line[:-1]] #converts text data to float and removes the last column (class label) from the data
        line.append(1) #add bias 1 to the input layer. Each line is now our "xi"
        x.append(line)
        y.append(yi)

#initialize parameters
w = np.subtract(np.random.rand(len(x[0]))*2,1) # weights between -1 and 1
lr = 0.015 #learning rate
threshold = 0.5 #for activation funciton
#convergence = 0.01 #i.e. stop training the weights once the difference in accuracy doesn't change by this much
accuracy = [] 
n_epoch = 0 

for epoch in range(1000):
    n_epoch +=1
    epoch_score = []
    y_preds = [] 
    grad_loss=[0]*len(x[0])

    for i in range(len(x)): #loop through data points 
        yhat = sigma(w,x[i]) #sigmoid activation function. 
        y_preds.append(yhat)
        
        #record whether prediction was right or wrong
        if (yhat > threshold and y[i] == 1) or (yhat < threshold and y[i] == 0): 
            epoch_score.append(1)
        else:
            epoch_score.append(0)

        #find the gradient of the loss function of L = (y-yhat)^2
        for j in range(len(x[i])): #loop through features of data point 
            grad_loss[j] += -2*(y[i]-sigma(w,x[i]))*sigma(w,x[i])*(1-sigma(w,x[i]))*x[i][j]

    #End of the epoch. 
    
    #update weights
    w = np.subtract(w, [lr * term for term in grad_loss])
    
    #record accuracy for this epoch 
    accuracy.append(sum(epoch_score)/len(epoch_score))
    

#at the end of training, print the last accuracy 
print("Accuracy:", accuracy[-1])
print("Runtime:", time.perf_counter() - start, "seconds")