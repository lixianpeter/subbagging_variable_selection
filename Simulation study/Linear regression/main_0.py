rep_ind_current=1
#!/usr/bin/env python
# coding: utf-8

# In[2]:


#rep_ind_current = 1
#!/usr/bin/env python
# coding: utf-8

# The following are the commonly used packages

import os
import numpy as np
import csv
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import time
import random 
import pandas as pd
import multiprocessing


# In[3]:


# Define some loss functions

def adaptive(beta,f,y,x,beta_0,lamda):
    return f(beta,y,x)+lamda*sum(abs(beta)/abs(beta_0))

# Numerical derivatives for testing
def first_derivative(f,beta,*args):
    grad=np.zeros(len(beta))
    i=0
    for coef in beta:
        h=np.zeros(len(beta))
        h[i]=1e-4
        grad[i]=(f(beta+h,*args)-f(beta,*args))/1e-4
        i+=1
    return grad
    
def second_derivative(f,beta,*args):
    hess=[]
    for i in range(0,len(beta)):
        h=np.zeros(len(beta))
        h[i]=1e-4
        hess+=[((first_derivative(f,beta+h,*args)-first_derivative(f,beta,*args))/1e-4)]
    return np.stack(hess)







# In[4]:


# For linear regression
def mse(beta,y,x):
    return sum((y-x@beta)**2)/len(y)
    
def mse_first_derivative(beta,y,x):
    return -2*(x.T@(y-x@beta))/len(y)

def mse_first_derivative_per_obs(beta, y, x):
    r = y - x @ beta                    
    return -2 * x * r[:, None]         
    
def mse_second_derivative(beta,y,x):
    return 2*x.T@x/len(y)

# For logistic regression
# def logistic(z):
#     return np.exp(z) / (1 + np.exp(z))
    
# def logistic_likelihood(beta,y,x):
#     X=x
#     Y=y
#     p=logistic(X@beta)
#     return -sum((Y*np.log(p)+(1-Y)*np.log(1-p)))/len(y)

# def logistic_first_derivative(beta,y,x):
#     Y=y
#     X=x
#     p=logistic(X@beta)
#     return -(Y@X-p@X)/len(y)

# def logistic_first_derivative_per_obs(beta, y, x):
#     X = x
#     Y = y
#     p = logistic(X @ beta)          
#     return X * (p - Y)[:, None]    
    
# def logistic_second_derivative(beta,y,x):
#     Y=y
#     X=x
#     p=logistic(X@beta)
#     return X.transpose()*(p*(1-p))@X/len(y)


# In[5]:


# This function is for a subsample's estimation

def subsample_estimate(subsample, f, beta_true = None): #  f is the loss; p is the dimension

    # extract one subsample
    #simu_data = pd.read_csv(file_name, header = 0).to_numpy()
    y_subsample = subsample[:,0]
    x_subsample = subsample[:,1:]
    p = x_subsample.shape[1]
    if (beta_true.all() == None):
        beta_true = np.zeros(p)
    k_N = len(y_subsample)

    
    # Obtain subsample estimates
    beta_subsample = np.linalg.inv(x_subsample.T@x_subsample)@x_subsample.T@y_subsample
    #minimize(f, beta_true, method = 'BFGS',    
                            #args = (y_subsample,x_subsample)).x
        # (x.tx)^(-1)xy
    # Obtain capital Sigma for linear regression
    second_derivative_subsample = mse_second_derivative(beta_subsample, y_subsample, x_subsample)

    # The following code is to prepare for the SE calculation
    # We need to use the correct first derivative for each observation, not the average derivative
    Sigma_hat_variance_subsample = mse_first_derivative_per_obs(beta_subsample, y_subsample, x_subsample).T @\
                                    mse_first_derivative_per_obs(beta_subsample, y_subsample, x_subsample)/k_N

    V_subsample = second_derivative_subsample


    return beta_subsample, second_derivative_subsample, Sigma_hat_variance_subsample, V_subsample





# In[6]:


# Define the least square approximation
def LSA(beta,beta_subsample,second_derivative_subsample,lamda=0):
    approx = 0
    m_N = len(beta_subsample)
    for i in range(0,m_N):
        #iterate through m_N subsamples
        approx += (beta-beta_subsample[i]).transpose()@second_derivative_subsample[i]@(beta-beta_subsample[i])
    approx = approx/m_N
    weights = np.mean(beta_subsample, axis=0)
    #weights[:]*= weights
    return approx + lamda*sum(abs(beta)/np.abs(weights)**2)


# In[7]:


# This function collects the information from each subsample in a list
def subbag(k_N, m_N, f, N, beta_true, e_noise = 1):
    # First, generate a full sample
    # Set true parameters
    # beta_true = np.concatenate([
    #     # setting boundaries to avoid true beta close to 0. The current set is 0.25
    #         np.r_[np.linspace(-1, -0.25, 6), np.linspace(0.25, 1, 6)],
    #         np.zeros(p - 12)])
    p = beta_true.shape[0]
    # # Create a csv file to store the simulated data
    # with open(file_name, mode='w',newline='') as file:
    #     f_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     header = ["y"] + [f"x{i}" for i in range(1, p + 1)]
    #     f_writer.writerow(header)
        
    # generate the p by p Toeplitz covariance matrix
    rho = 0.5
    idx = np.arange(p)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    
    # Generate correlated covariates
    x = np.random.multivariate_normal(
    mean=np.zeros(p),
    cov=Sigma,
    size=N
    )   # shape: (N, p)

    # Generate all noise at once
    eps = np.random.normal(
    loc=0,
    scale=e_noise**0.5, # Note its SD not variance
    size=N
    )   # shape: (N,)

    y = x @ beta_true + eps

    simu_data =  np.hstack((y[:, None], x))
    # Create lists to collect the information
    beta_subsample_list = []
    second_derivative_subsample_list =[]
    Sigma_hat_variance_subsample_list = []
    V_subsample_list = []
    
    for i in range(0,m_N):
        subsample = simu_data[random.sample(range(1,len(simu_data)), k = k_N)]
        # the result from the above function
        #result = subsample_estimate(subsample = subsample, f = f, beta_true = beta_true)
        y_subsample = subsample[:,0]
        x_subsample = subsample[:,1:]
        
        # Obtain subsample estimates
        beta_subsample = np.linalg.inv(x_subsample.T@x_subsample)@x_subsample.T@y_subsample
        #minimize(f, beta_true, method = 'BFGS',    
                                #args = (y_subsample,x_subsample)).x
            # (x.tx)^(-1)xy
        # Obtain capital Sigma for linear regression
        second_derivative_subsample = mse_second_derivative(beta_subsample, y_subsample, x_subsample)
    
        # The following code is to prepare for the SE calculation
        # We need to use the correct first derivative for each observation, not the average derivative
        Sigma_hat_variance_subsample = mse_first_derivative_per_obs(beta_subsample, y_subsample, x_subsample).T @\
                                        mse_first_derivative_per_obs(beta_subsample, y_subsample, x_subsample)/k_N 
        
        beta_subsample_list += [beta_subsample]
        second_derivative_subsample_list += [second_derivative_subsample]
        Sigma_hat_variance_subsample_list += [Sigma_hat_variance_subsample]
        V_subsample_list += [second_derivative_subsample]
    return beta_subsample_list, second_derivative_subsample_list, Sigma_hat_variance_subsample_list, V_subsample_list



# In[8]:


# Define the function that selects the best lambda
def SBIC(k_N, m_N, result, initial_value, lamda_constant = 1, interval = 0.0000001, scale = True):
    BIC_min = float('inf')
    for log_scale in range(0, int(-np.log10(interval))):
        lamda = lamda_constant * 10 ** (-log_scale)
        alpha = (k_N * m_N)/N
        estimate = minimize(LSA, initial_value, method = 'Powell', args = (result[0], result[1], lamda)).x
        df = sum(abs(estimate) > 10e-16)
        if scale == True:
            BIC = k_N * LSA(estimate, result[0], result[1], lamda = lamda) + df * np.log(N)
        if scale == False:
            BIC = LSA(estimate, result[0], result[1],lamda = lamda) + df * np.log(N)
        if BIC < BIC_min:
            BIC_min = BIC
            lamda_min = lamda
            estimate_optimal = estimate
    return BIC_min, lamda_min, estimate_optimal


# In[9]:


# # Linear Regression

# N = 500000
# beta = np.array([3,1.5,0,0,2,0,0,0])
# with open('sim linear data_N=500000_2.csv', mode='w',newline='') as file:
#     f_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     f_writer.writerow(["y","x1","x2","x3","x4","x5","x6","x7","x8"])
#     for n in range(0,N):
#         x = np.random.normal(0,1,8)
#         y = x@beta + np.random.normal(0,1,1)
#         f_writer.writerow(y.tolist()+x.tolist())
# file.close()
# simu_data=np.genfromtxt('sim linear data_N=500000_2.csv', delimiter=',')
# y=simu_data[1:,0]
# x=simu_data[1:,1:]
# beta_OLS=minimize(mse, beta, method="Powell",args=(y,x)).x
# beta_adaptive=minimize(adaptive, np.array([3,1.5,0,0,2,0,0,0]), method="Powell",args=(mse,y,x,beta_OLS,0.01)).x
# beta_adaptive








# In[ ]:


def sim_saver(k_N, m_N, N, e_noise = 1, p = 200): # e_noise is the variance of error term in linear regression
    alpha=(k_N * m_N)/N
    beta_true = np.concatenate([
            np.r_[np.linspace(-1, -0.5, 6), np.linspace(0.5, 1, 6)],
            np.zeros(p - 12)])
    # # Covariance matrix
    # rho = 0.5
    # idx = np.arange(p)
    # Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    # SNR = beta_true.T@Sigma@beta_true/e_noise
    # prepare writing for subsample results
    # If the summary file does not exist, create a new one
    file_name = '../result/N=' + str(N) + '_k_N='+str(k_N)+'_'+'m_N='+str(m_N)+'_'+'p='+str(p)+'_e_noise=' +\
                str(e_noise)+            '_.csv'
    if (not (os.path.exists(file_name))):
        with open(file_name, mode='w',newline='') as f:
            f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = ["index"]
            
            # lasso betas
            #header += [f"lasso beta_{i}" for i in range(1, p + 1)]
            
            # lasso true betas
            header += [f"lasso_true beta_{i}" for i in range(1, p + 1)]
            
            # subsample SEs (only beta indices you want)
            subsample_betas = list(range(1,13))
            #for k in range(1, 6):   # SE1 ... SE5
            header += [f"subsample SE beta_{i}" for i in subsample_betas]
            
            # CIs
            #for k in range(1, 6):   # CI1 ... CI5
            header += [f"CI beta_{i}" for i in subsample_betas]
            
            # scalars
            header += [
                #"BIC_min", "lamda_min",
                "BIC_min_true", "lamda_min_true",
                "time1", "time2",
                "memory"
            ]
            f_writer.writerow(header)
            
    # Simulation start writing into the corresponding files            
    with open(file_name, mode = 'a',newline = '') as f:
        
        f_writer = csv.writer(f, delimiter = ',', quotechar = '"', quoting=csv.QUOTE_MINIMAL)
        
        for i in range(0,10):
            
            random.seed(rep_ind_current+i)
            np.random.seed(rep_ind_current+i)
            
            start_time = time.time()
            # obtain the collection from subbag files
            result = subbag(#'sim data_N=' + str(N) + '_' + str(rep_ind_current+i) + '_p=' + str(p) + '.csv',
                            k_N, m_N, mse, N, beta_true, e_noise)
            end_time = time.time()
            
            # Simple average of subbagging estimates
            estimate = np.mean(result[0], axis = 0)

            # start_time1 = time.time()
            # # LSA minmizer; we set lambda small to avoid potential bias for now
            # # The optimizer method Powerll can give exactly value of 0 when intial value is true beta
            # # For time comparsion to the tuning parameter of lambda
            # estimate_lasso = minimize(LSA, beta_true, method='Powell',args=(result[0],result[1],0.00001)).x
            # end_time1 = time.time()

            # Lasso uses subbagging average as initial value
            start_time2 = time.time()
            lasso_result = SBIC(k_N, m_N, result, initial_value = beta_true)
            estimate_lasso_true = lasso_result[2]
            BIC_min_true = lasso_result[0]
            lamda_min_true = lasso_result[1]
            end_time2 = time.time()

            # Sandwich matrix SE calculation
            SE1_subsample = np.sqrt(((1 + 1/alpha)/N * np.linalg.inv(np.mean(result[3], axis = 0)[:12, :12]).T @ \
                        np.mean(result[2], axis = 0)[:12, :12] @\
                        np.linalg.inv(np.mean(result[3], axis = 0)[:12, :12]))[np.arange(12), np.arange(12)])

            
            # Coverage of confidence interval based on the SE
            CI1_subsample = (estimate_lasso_true[:12] + norm.ppf(0.975) * SE1_subsample > beta_true[0:12]) * (estimate_lasso_true[:12] - norm.ppf(0.975) * SE1_subsample < beta_true[0:12])
            
            f_writer.writerow(([rep_ind_current+i]) + 
                              estimate_lasso_true.tolist() +
                              SE1_subsample.tolist() +
                              CI1_subsample.astype(int).tolist() +
                              ([BIC_min_true]) +
                              ([lamda_min_true]) +
                              #[end_time1 - start_time1 + end_time - start_time] +
                              [end_time2 - start_time2 + end_time - start_time] 
                             )





# In[38]:


# Signal-noise-ratio
p=12
beta_true = np.concatenate([
        np.r_[np.linspace(-1, -0.5, 6), np.linspace(0.5, 1, 6)],
        np.zeros(p - 12)])
rho = 0.5
idx = np.arange(p)
Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
beta_true.T@Sigma@beta_true

# For ratio = 2
beta_true.T@Sigma@beta_true/2

# For ratio = 0.5
beta_true.T@Sigma@beta_true/0.5


# In[ ]:


# # Linear regression
# # For test only
N = 50000
alpha = 1
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N, e_noise = 1, p =15)
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N, e_noise = 1, p =15)
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N, e_noise = 1, p =15)

N = 500000
alpha = 0.5
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/2, p = 30)
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/0.5, p = 30)

sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/2, p = 30)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/0.5, p = 30)





alpha = 1
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/2, p = 30)
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/0.5, p = 30)

sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/2, p = 30)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/0.5, p = 30)






# In[ ]:


N = 1000000
alpha = 0.5
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/2, p = 30)
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/0.5, p = 30)

sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/2, p = 30)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/0.5, p = 30)





alpha = 1
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/2, p = 30)
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/0.5, p = 30)

sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/2, p = 30)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N, e_noise = beta_true.T@Sigma@beta_true/0.5, p = 30)



# In[ ]:





# In[ ]:




