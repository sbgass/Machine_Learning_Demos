#!/usr/bin/env python3
import numpy as np
from scipy.stats import multivariate_normal
if not __file__.endswith('_em_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as LastName_em_gaussian.py (replacing LastName with your last name)!')
    exit(1)

DATA_PATH = "/u/cs246/data/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

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
        lambdas = np.zeros(args.cluster_num) #lambda is height of each gaussian distribution. Technically, p(z=k)=lambda_k
        mus = np.zeros((args.cluster_num,2)) #mu is the mean coordinates (x,y) of each cluster 
        if not args.tied:
            sigmas = np.array([np.identity(2)]*args.cluster_num) #np.ones((args.cluster_num,2,2))
        else:
            sigmas = np.identity(2) #sigma is the covariance matrix 2x2 for 2D space. S=([s(xx),s(xy)],[s(yx),s(yy)])
        
        #Randomly initialize lambdas, mus, and sigmas 
        lambdas = np.random.rand(*lambdas.shape)
        mus = np.random.rand(*mus.shape)
    else:
        lambdas = []
        mus = []
        sigmas = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #lambda mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float,line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2])
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]])
        lambdas = np.asarray(lambdas)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)

    model = (lambdas, mus, sigmas)
    return model

def get_norm_constant(model, xn,args):
    if args.tied:
        lambdas, mus, sigmas = model 
        C = sum([lambda_k*multivariate_normal(mean=mu_k,cov=sigmas).pdf(xn) for lambda_k,mu_k in zip(lambdas,mus)])
    else:
        C = sum([lambda_k*multivariate_normal(mean=mu_k,cov=sigma_k).pdf(xn) for lambda_k,mu_k,sigma_k in zip(*model)])
    return C 

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #NOTE: you can use multivariate_normal like this:
    #probability_of_xn_given_mu_and_sigma = multivariate_normal(mean=mu, cov=sigma).pdf(xn)
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
    lambdas, mus, sigmas = model 

    #print(f"lambads: {lambdas}")
    #print(f"mus: {mus}")
    #print(f"sigmas: {sigmas}")
    
    if args.cluster_num:
        k = args.cluster_num
    else:
        k = 2 

    for epoch in range(args.iterations):
        z = np.zeros((len(train_xs),k)) #this is our P(x|i) for every data point and every class
        #find the class probability distribution for each data point 
        for n in range(len(train_xs)): 
            norm_constant = get_norm_constant((lambdas,mus,sigmas),train_xs[n],args)
            for i in range(k):
                if args.tied:
                    z[n,i] = lambdas[i] * multivariate_normal(mean=mus[i],cov=sigmas).pdf(train_xs[n])/norm_constant
                else:
                    z[n,i] = lambdas[i] * multivariate_normal(mean=mus[i],cov=sigmas[i]).pdf(train_xs[n])/norm_constant
                
        #update clusters to the centers of the data 
        new_sigma = np.zeros_like(sigmas)
        for i in range(k):
            zin_sum = sum([z[n,i] for n in range(len(train_xs))])
            lambdas[i] = zin_sum/len(train_xs)
            mus[i] = sum([z[n,i]*train_xs[n] for n in range(len(train_xs))])/zin_sum
            if args.tied:
                new_sigma += sum([z[n,i]*np.dot((train_xs[n]-mus[i])[np.newaxis].T,(train_xs[n]-mus[i])[np.newaxis]) for n in range(len(train_xs))])/zin_sum/k #average the variances avroce the clusters 
            else:
                sigmas[i] = sum([z[n,i]*np.dot((train_xs[n]-mus[i])[np.newaxis].T,(train_xs[n]-mus[i])[np.newaxis]) for n in range(len(train_xs))])/zin_sum
            #^slicing x-mu with np.newaxis makes the 1d vectors into 2d (with dimension (1,k)), so you can transpose. np won't transpose a 1d vector... 
            #^Also, 2x1 dot 1x2 = 2x2, which is the dim of cov matrix 

        if args.tied:
            sigmas = new_sigma 

    model = (lambdas, mus, sigmas)
    #print("\nUpdated:")
    #print(f"lambads: {lambdas}")
    #print(f"mus: {mus}")
    #print(f"sigmas: {sigmas}")
    return model

def average_log_likelihood(model, data, args):
    from math import log
    from scipy.stats import multivariate_normal
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    ll = 0.0
    #objective function of EM: max l,m,s for log ( PI ( lambda_zn * N(xn,mu_zn,sigma_zn) ) )
    ## Note that lambda*N(x;m,s) = P(k)*P(x|k) = P(x)  
    
    lambdas, mus, sigmas = model 

    if args.cluster_num:
        k = args.cluster_num
    else:
        k = 2 

    #find which cluster each data point belongs in: find all probabilities, argmax for each data point 
    z = np.zeros((len(data),k)) #this is our P(x|i) for every data point and every class
    for n in range(len(data)): 
        norm_constant = get_norm_constant((lambdas,mus,sigmas),data[n],args)
        for i in range(k):
            if args.tied:
                z[n,i] = lambdas[i] * multivariate_normal(mean=mus[i],cov=sigmas).pdf(data[n]) #/ norm_constant
            else: 
                z[n,i] = lambdas[i] * multivariate_normal(mean=mus[i],cov=sigmas[i]).pdf(data[n]) #/ norm_constant

        ll += np.log(sum(z[n])) #average log likelihoods of all the data points 
    
    ll = ll/len(data)
    #print(f"log likelihood: {ll}")
    
    return ll

def extract_parameters(model):
    return model

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points.')
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
    ll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()
