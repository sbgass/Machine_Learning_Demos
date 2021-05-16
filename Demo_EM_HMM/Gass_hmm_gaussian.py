#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_hmm_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
    exit(1)

DATA_PATH = "/u/cs446/data/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    if args.cluster_num:
        mus = np.zeros((args.cluster_num,2))
        if not args.tied:
            sigmas = np.array([np.identity(2)]*args.cluster_num)
        else:
            sigmas = np.identity(2) 
        transitions = np.zeros((args.cluster_num,args.cluster_num)) #transitions[i][j] = probability of moving from cluster i to cluster j
        initials = np.zeros(args.cluster_num) #probability for starting in each state
        
        #Random values 
        initials = np.ones_like(initials)
        initials = initials/initials.sum()
        mus = np.random.rand(*mus.shape)
        #no lambda...? 
        transitions = np.random.rand(*transitions.shape)
    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float,line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5],vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)       #k where k is the number of clusters 
        transitions = np.asarray(transitions) #k x k
        mus = np.asarray(mus)                 #k x 2 
        sigmas = np.asarray(sigmas)           #k x 2 x 2 
        args.cluster_num = len(initials)

    model = (initials, transitions, mus, sigmas)
    return model

def forward(model, data, args):
    from scipy.stats import multivariate_normal
    from math import log
    alphas = np.zeros((len(data),args.cluster_num))
    log_likelihood = 0.0
    #TODO: Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
    #NOTE: To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value, and increment log_likelihood by the log of the value you normalized by. This will prevent the probabilities from going to 0, and the scaling will be cancelled out in train_model when you normalize (you don't need to do anything different than what's in the notes). This was discussed in class on April 8th. 

    initials, transitions, mus, sigmas = model 
    k = args.cluster_num 
    emissions = np.zeros((len(data),k))

    #rint(f"initials:{initials.shape}\ntransitions:{transitions.shape}\nmus:{mus.shape}\nsigmas:{sigmas.shape}")
    #rint()

    #create alphas by looping through data points 
    for n in range(len(data)):
        #build emissions matrix
        for i in range(k):
            if args.tied:
                emissions[n][i] = multivariate_normal(mean=mus[i],cov=sigmas).pdf(data[n])
            else:
                emissions[n][i] = multivariate_normal(mean=mus[i],cov=sigmas[i]).pdf(data[n])  
            
        #for the starting states: n=0  
        if n == 0: 
            #alphas[n] = np.dot(transitions.T, initials) * emissions[n] #element wise multiplication for emissions
            alphas[n] = initials.copy()
        else:
            alphas[n] = np.dot(transitions.T, alphas[n-1]) * emissions[n] #element wise multiplication for emissions
        #^(kxk) dot kx1 * kx1 = kx1 
        
        #Normalize alpha across all clusters
        denom = alphas[n].sum() 
        alphas[n] = alphas[n]/denom

        #increment log likelihood
        log_likelihood += log(denom) 

    log_likelihood/=len(data)

    return alphas, log_likelihood, emissions

def backward(model, data, args):
    from scipy.stats import multivariate_normal
    betas = np.zeros((len(data),args.cluster_num))
    #TODO: Calculate and return backward probabilities (normalized like in forward before)
    
    initials, transitions, mus, sigmas = model 
    k = args.cluster_num 
    emissions = np.zeros((len(data),k))

    #from back to front 
    for n in reversed(range(len(data))):
        #build emissions matrix
        for i in range(k):
            if args.tied:
                emissions[n][i] = multivariate_normal(mean=mus[i],cov=sigmas).pdf(data[n])
            else:
                emissions[n][i] = multivariate_normal(mean=mus[i],cov=sigmas[i]).pdf(data[n]) 
        
        #for the ending point: 
        if n == len(data)-1:
            #betas[n] = np.dot(transitions, np.ones(k)) * emissions[n]
            betas[n] = np.ones(k)
        else:
            betas[n] = np.dot(transitions, betas[n+1]) * emissions[n]
        
        #Normalize beta 
        betas[n] = betas[n]/betas[n].sum()

    return betas

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
    
    initials, transitions, mus, sigmas = model 
    k = args.cluster_num 
    dev_ll = np.zeros(args.iterations) #maximum number of dev log likelihoods will be the max number of iterations 
    emissions = np.zeros((len(train_xs),k)) # capital "B" in his lecture notes 
    gammas = np.zeros((len(train_xs),k)) #n by k. This is used to update mu and sigma
    xis = np.zeros((len(train_xs)-1,k,k)) #n-1 x k x k (This is the probability of each possible transition given our data and our model. There are only n-1 transitions.)  

    for epoch in range(args.iterations):

        ## E-step: run forward/backward algorithm to get alpha and betas, then calculate gammas and xis. 
        alphas, log_likelihood, emissions = forward((initials,transitions,mus,sigmas), train_xs,args)
        betas = backward((initials,transitions,mus,sigmas),train_xs,args)
        
        #calculate gammas (n by k) with each new alpha and beta probabilities
        for n in range(len(train_xs)):
            denom = np.multiply(alphas[n],betas[n]).sum()  #element wise multiplication then a sum to get a scalar 
            gammas[n] = np.multiply(alphas[n],betas[n])/denom

        #calculate xis (n by k by k) with alpha, beta, A, and B
        for n in range(1, len(train_xs)):
            denom = sum([sum([alphas[n-1,i]*emissions[n,j]*betas[n,j]*transitions[j,i]  for j in range(k)]) for i in range(k)])
            for i in range(k):
                for j in range(k): 
                    xis[n-1,i,j] = (alphas[n-1,i] * emissions[n,j] * betas[n,j] * transitions[j,i])/denom #equation in notes
                    
        ## M-step: Update A, mus, and sigmas using xis and gammas 
        #update "A": the transition matrix 
        for i in range(k):
            denom = np.sum(np.sum(xis[:,i,:],axis=1),axis=0)
            for j in range(k):
                transitions[i,j] = np.sum(xis[:,i,j],axis=0)/denom
        
        #update "B": the emissions model 
        denom = np.sum(gammas,axis=0) #axis=0 is summing over n. This result is an kx1 vector.
        sigmas = np.zeros((2,2)) if args.tied else np.zeros((k,2,2))#initialize sigmas to be updated 
        for i in range(k):
            mus[i] = sum([gammas[n,i]*train_xs[n] for n in range(len(train_xs))])/denom[i] #train_xs[n] has shape (2,)
            #mus will be k x 2 

            if args.tied:
                sigmas += np.sum([gammas[n,i]*np.dot((train_xs[n]-mus[i])[np.newaxis].T,(train_xs[n]-mus[i])[np.newaxis]) for n in range(len(train_xs))],axis=0)/denom[i]/k#average the variances over the clusters
            else:
                sigmas[i] = np.sum([gammas[n,i]*np.dot((train_xs[n]-mus[i])[np.newaxis].T,(train_xs[n]-mus[i])[np.newaxis]) for n in range(len(train_xs))],axis=0)/denom[i] 
        #^ sigmas will be k x 2 x 2

        #update initial alphas at timestep n=0 
        initials = gammas[0].copy() 


        #use dev data to control iterations 
        if not args.nodev and epoch > 0:
            a, dev_ll[epoch], e = forward((initials,transitions,mus,sigmas), train_xs,args)
            
            if dev_ll[epoch] - dev_ll[epoch-1] < 0: #if ll doesn't improve from one iteration to the next
                model = (initials, transitions, mus, sigmas)
                return model 


    model = (initials, transitions, mus, sigmas)
    return model

def average_log_likelihood(model, data, args):
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    #NOTE: yes, this is very simple, because you did most of the work in the forward function above
    ll = 0.0
    
    a, ll, e = forward(model,data,args)

    return ll

def extract_parameters(model):
    initials = model[0]
    transitions = model[1]
    mus = model[2]
    sigmas = model[3]
    return initials, transitions, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    nll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(nll_train))
    if not args.nodev:
        nll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(nll_dev))
    initials, transitions, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '),transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()
