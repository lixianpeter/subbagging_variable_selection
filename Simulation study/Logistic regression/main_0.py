rep_ind_current = 1

#!/usr/bin/env python
# coding: utf-8

# In[164]:


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


# In[165]:


# define some loss functions

def adaptive(beta,f,y,x,beta_0,lamda):
    return f(beta,y,x)+lamda*sum(abs(beta)/abs(beta_0))

# def var_of_mse_grad_linear(x, y, beta):
#     n, p =x.shape
#     r = y - x @ beta                         # residuals, (n,)
#     g_i = -2 * x * r[:, None]/len(y)              # (n,p), each row is g_i(beta)

#     gbar = g_i.mean(axis=0)                  # (p,)
#     centered = g_i - gbar[None, :]           # (n,p)

#     cov_gi = centered.T @ centered / (n - 1) # (p,p) sample covariance of g_i
#     return cov_gi

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







# In[166]:


# For linear regression
# def mse(beta,y,x):
#     return sum((y-x@beta)**2)/len(y)
    
# def mse_first_derivative(beta,y,x):
#     return -2*(x.T@(y-x@beta))/len(y)
    
# def mse_second_derivative(beta,y,x):
#     return 2*x.T@x/len(y)

# For logistic regression
def logistic(z):
    return np.exp(z) / (1 + np.exp(z))
    
def logistic_likelihood(beta,y,x):
    X=x
    Y=y
    p=logistic(X@beta)
    return -sum((Y*np.log(p)+(1-Y)*np.log(1-p)))/len(y)

def logistic_first_derivative(beta,y,x):
    Y=y
    X=x
    p=logistic(X@beta)
    return -(Y@X-p@X)/len(y)
    
def logistic_second_derivative(beta,y,x):
    Y=y
    X=x
    p=logistic(X@beta)
    return X.transpose()*(p*(1-p))@X/len(y)


# In[167]:


# This function is for a subsample's estimation

def subsample_estimate(file_name,subsize,f,p = 200): # subsize is k_N; f is the loss

    # extract one subsample
    simu_data = pd.read_csv(file_name, header = 0).to_numpy()
    subsample = simu_data[random.sample(range(1,len(simu_data)), k = subsize)]
    y_subsample = subsample[:,0]
    x_subsample = subsample[:,1:]
    beta_true = np.concatenate([
            np.linspace(-1, 1, 12),
            np.zeros(p - 12)])
    
    # obtain subsample estimates
    beta_subsample = minimize(f, beta_true, method = 'BFGS',    
                            args = (y_subsample,x_subsample)).x

    #if(f == mse):
    # obtain captial Sigma for linear regession
    # second_derivative_subsample = mse_second_derivative(beta_subsample, y_subsample, x_subsample)

    # obtain captial Sigma for logistic regession
    second_derivative_subsample = logistic_second_derivative(beta_subsample, y_subsample, x_subsample)

    # obtain the middle matrix in the sandwich covariance matrix (using true beta or estimate)
    # first_derivative_subsample = mse_first_derivative(beta_subsample, y_subsample, x_subsample)
    # first_derivative_true =  mse_first_derivative(beta_true, y_subsample, x_subsample) 

    # Linear regression
    # Sigma_hat_variance_subsample = np.sum( (y_subsample - x_subsample @ beta_subsample)**2 )  / (len(y_subsample))
    # Sigma_hat_variance_true = np.sum( (y_subsample - x_subsample @ beta_true)**2 )  / (len(y_subsample))
    # # new subsample variance
    # V_Sigma_V = Sigma_hat_variance_subsample * np.linalg.inv(x_subsample.T @ x_subsample / len(y_subsample))

    # Logistic regression
    # with subsample estimate
    # p = logistic(x_subsample@beta_subsample)          
    # r = (p - y_subsample)                      
    # G =  x_subsample * r[:, None]               # (m, p), rows are g_i^T                
    # Sigma_hat_variance_subsample = G.T @ G / len(y_subsample) # (p, p) = Σ g_i g_i^T
    # with true beta
    # p = logistic(x_subsample@beta_true)          
    # r = (p - y_subsample)                      
    # G =  x_subsample * r[:, None]               # (m, p), rows are g_i^T  
    Sigma_hat_variance_subsample = np.outer(logistic_first_derivative(beta_subsample, y_subsample, x_subsample),
                                                        logistic_first_derivative(beta_subsample, y_subsample, x_subsample)) / len(y_subsample)

    V_subsample = second_derivative_subsample
    #V_true = logistic_second_derivative(beta_true, y_subsample, x_subsample)
    
    

    return beta_subsample, second_derivative_subsample, Sigma_hat_variance_subsample, V_subsample





# In[168]:


# define the least square approximation
def LSA(beta,beta_subsample,second_derivative_subsample,lamda=0):
    approx=0
    m_N=len(beta_subsample)
    for i in range(0,m_N):
        #iterate through m_N subsamples
        approx += (beta-beta_subsample[i]).transpose()@second_derivative_subsample[i]@(beta-beta_subsample[i])
    approx = approx/m_N
    weights = np.mean(beta_subsample, axis=0)
    #weights[:]*= weights
    return approx + lamda*sum(abs(beta)/np.abs(weights)**2)


# In[169]:


# This function collect the information from each subsample in a list

def subbag(file_name, k_N, m_N, f, N, p = 200):
    # First generate full sample
    beta_true = np.concatenate([
            np.linspace(-1, 1, 12),
            np.zeros(p - 12)])
    with open(file_name, mode='w',newline='') as file:
        f_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ["y"] + [f"x{i}" for i in range(1, p + 1)]
        f_writer.writerow(header)
        
        # generate the Toeplitz covariance matrix
        rho = 0.5
        idx = np.arange(p)
        Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
        for n in range(0,N):
            x = np.random.multivariate_normal(
                mean=np.zeros(p),
                cov=Sigma
            )
            # logistic model: P(y=1|x) = sigmoid(x @ beta)
            z = x @ beta_true
            prob = logistic(z)
            y = np.random.binomial(1, prob, 1)  # shape (1,)
            f_writer.writerow(y.tolist()+x.tolist())
    file.close()

    # create lists to collect the information
    beta_subsample = []
    second_derivative_subsample =[]
    Sigma_hat_variance_subsample = []
    #Sigma_hat_variance_true = []
    V_subsample = []
    #V_true = []
    for i in range(0,m_N):
        # the result from the above function
        result = subsample_estimate(file_name = file_name, subsize = k_N, f=f, p = p)
        beta_subsample += [result[0]]
        second_derivative_subsample += [result[1]]
        Sigma_hat_variance_subsample += [result[2]]
        #Sigma_hat_variance_true += [result[3]]
        V_subsample += [result[3]]
        #V_true += [result[5]]
    # Delete the file
    os.remove(file_name)
    return beta_subsample, second_derivative_subsample, Sigma_hat_variance_subsample, V_subsample



# In[170]:


# define the function that selects the best lambda
def SBIC(k_N, m_N, result, initial_value, lamda_constant = 5, interval = 0.0000001, scale = True):
    BIC_min = float('inf')
    # beta_true = np.array([3,1.5,0,0,2,0,0,0])
    # beta_subbagging_average = np.mean(result[0], axis = 0)
    for log_scale in range(0, int(-np.log10(interval))):
        lamda = lamda_constant * 10 ** (-log_scale)
        alpha = (k_N * m_N)/N
        estimate = minimize(LSA, initial_value, method = "Powell", args = (result[0], result[1], lamda)).x
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


# In[171]:


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



# beta_true = np.concatenate([
#             np.linspace(-1, 1, 12),
#             np.zeros(p - 12)])






# In[172]:


def sim_saver(k_N, m_N, N, p = 200):
    alpha=(k_N * m_N)/N
    beta_true = np.concatenate([
            np.linspace(-1, 1, 12),
            np.zeros(p - 12)])
    # prepare writing for subsample results
    # SE_fullsample=np.sqrt((1+1/alpha)*np.linalg.inv(second_derivative(mse,beta,y,x)[[0,1,4],:][:,[0,1,4]])[[0,1,2],[0,1,2]])
    # If summary file not exist, create a new one
    file_name = '../result/N=' + str(N) + '_k_N='+str(k_N)+'_'+'m_N='+str(m_N)+'_'+'.csv'
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
            
    # simulation start writing into the corresponding files            
    with open(file_name, mode = 'a',newline = '') as f:
        
        f_writer = csv.writer(f, delimiter = ',', quotechar = '"', quoting=csv.QUOTE_MINIMAL)
        
        for i in range(0,2):
            
            random.seed(rep_ind_current+i)
            
            start_time = time.time()
            # obtain the collection from subbag files
            result = subbag('sim data_N=' + str(N) + '_' + str(rep_ind_current+i) + '_p=' + str(p) + '.csv',
                            k_N, m_N, logistic_likelihood, N, p)
            end_time = time.time()
            
            # Simple average of subbagging estimates
            estimate = np.mean(result[0], axis = 0)

            start_time1 = time.time()
            # LSA minmizer; we set lambda small to avoid potential bias for now
            # The optimizer method Powerll can give exactly value of 0 when intial value is true beta
            # For time comparsion to the tuning parameter of lambda
            estimate_lasso = minimize(LSA, beta_true, method='Powell',args=(result[0],result[1],0.00001)).x
            end_time1 = time.time()

            # Lasso uses subbagging average as initial value
            start_time2 = time.time()
            lasso_result = SBIC(k_N, m_N, result, initial_value = beta_true)
            estimate_lasso_true = lasso_result[2]
            BIC_min_true = lasso_result[0]
            lamda_min_true = lasso_result[1]
            end_time2 = time.time()

            # First kind of SE calculation; i.e., based on the subbagging
            #SE1_subsample = np.sqrt(k_N * (1 + 1/alpha) * ((np.array(result[0]) - estimate).T@(np.array(result[0]) - estimate))[[0,1,4],[0,1,4]]/m_N/N)

            # Other sandwitch matrix SE calculation
            # SE1_subsample = np.sqrt((1 + 1/alpha)/N * np.linalg.inv(np.mean(result[2], axis = 0)).T @ \
            #                         np.mean(result[4], axis = 0) @\
            #                         np.linalg.inv(np.mean(result[4], axis = 0)))[[0,1,4],[0,1,4]]
            SE1_subsample = np.sqrt(((1 + 1/alpha)/N * np.linalg.inv(np.mean(result[1], axis = 0)[:12, :12]).T @ \
                        np.mean(result[2], axis = 0)[:12, :12] @\
                        np.linalg.inv(np.mean(result[2], axis = 0)[:12, :12]))[np.arange(12), np.arange(12)])
            # SE4_subsample = np.sqrt((1 + 1/alpha)/N * np.linalg.inv(np.mean(result[3], axis = 0)).T @ \
            #                         np.mean(result[5], axis = 0) @\
            #                         np.linalg.inv(np.mean(result[3], axis = 0)))[[0,1,4],[0,1,4]]
            # SE5_subsample = np.sqrt((1 + 1/alpha)/N * np.linalg.inv(np.mean(result[3], axis = 0)[np.ix_([0,1,4],[0,1,4])]).T @ \
            #                         np.mean(result[5], axis = 0)[np.ix_([0,1,4],[0,1,4])] @\
            #                         np.linalg.inv(np.mean(result[3], axis = 0))[np.ix_([0,1,4],[0,1,4])])[[0,1,2],[0,1,2]]
            # # New SE calculation
            #SE6_subsample = np.sqrt((1 + 1/alpha)/N * np.mean(result[4], axis=0)[[0,1,4],[0,1,4]])

            
            # Coverage of confidence interval based on the SE
            CI1_subsample = (estimate_lasso_true[:12] + norm.ppf(0.975) * SE1_subsample > np.linspace(-1, 1, 12)) * (estimate_lasso_true[:12] - norm.ppf(0.975) * SE1_subsample < np.linspace(-1, 1, 12))
            # CI2_subsample = (estimate_lasso_true[[0,1,4]] + norm.ppf(0.975) * SE2_subsample > [3, 1.5, 2]) * (estimate_lasso_true[[0,1,4]] - norm.ppf(0.975) * SE2_subsample < [3, 1.5, 2])
            # CI3_subsample = (estimate_lasso_true[[0,1,4]] + norm.ppf(0.975) * SE3_subsample > [3, 1.5, 2]) * (estimate_lasso_true[[0,1,4]] - norm.ppf(0.975) * SE3_subsample < [3, 1.5, 2])            
            # CI4_subsample = (estimate_lasso_true[[0,1,4]] + norm.ppf(0.975) * SE4_subsample > [3, 1.5, 2]) * (estimate_lasso_true[[0,1,4]] - norm.ppf(0.975) * SE4_subsample < [3, 1.5, 2])            
            # CI5_subsample = (estimate_lasso_true[[0,1,4]] + norm.ppf(0.975) * SE5_subsample > [3, 1.5, 2]) * (estimate_lasso_true[[0,1,4]] - norm.ppf(0.975) * SE5_subsample < [3, 1.5, 2])            
            # CI6_subsample = (estimate_lasso[[0,1,4]] + norm.ppf(0.975) * SE6_subsample > [3, 1.5, 2]) * (estimate[[0,1,4]] - norm.ppf(0.975) * SE6_subsample < [3, 1.5, 2])            
            # CI7_subsample = (estimate[[0,1,4]] + 1.96 * SE2_subsample > [3, 1.5, 2]) * (estimate[[0,1,4]] - 1.96 * SE2_subsample < [3, 1.5, 2])            

            
            f_writer.writerow(([rep_ind_current+i]) + 
                              #estimate.tolist() + 
                              #estimate_lasso.tolist() +
                              estimate_lasso_true.tolist() +
                              SE1_subsample.tolist() +
                              # SE2_subsample.tolist() + 
                              # SE3_subsample.tolist() +
                              # SE4_subsample.tolist() +
                              # SE5_subsample.tolist() +
                              #SE6_subsample.tolist() +
                              CI1_subsample.astype(int).tolist() +
                              # CI2_subsample.astype(int).tolist() +
                              # CI3_subsample.astype(int).tolist() +
                              # CI4_subsample.astype(int).tolist() +
                              # CI5_subsample.astype(int).tolist() +
                              #CI6_subsample.astype(int).tolist() +
                              #CI7_subsample.astype(int).tolist() +
                              # ([BIC_min]) +
                              # ([lamda_min]) +
                              ([BIC_min_true]) +
                              ([lamda_min_true]) +
                              [end_time1 - start_time1 + end_time - start_time] +
                              [end_time2 - start_time2 + end_time - start_time] 
                             )





# In[174]:


# Logistics regression
N = 50000
alpha = 1
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N, p =15)

N = 500000
alpha = 0.2
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)




alpha = 0.5
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)





alpha = 1
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)



alpha = 2
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)





# In[ ]:


N = 1000000
alpha = 0.2
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)




alpha = 0.5
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)





alpha = 1
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)



alpha = 2
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)





# In[ ]:


# Logistic regression
# N = 500000
# alpha = 0.2
# sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
# sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)




# alpha = 0.5
# sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
# sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)





# alpha = 1
# sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
# sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)



# alpha = 2
# sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
# sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# # Logistic regression

# In[ ]:


# def logistic_first_derivative(beta,y,x):
#     Y=y
#     X=x
#     p=logistic(X@beta)
#     return -(Y@X-p@X)
# def logistic_second_derivative(beta,y,x):
#     Y=y
#     X=x
#     p=logistic(X@beta)
#     return X.transpose()*(p*(1-p))@X


# In[ ]:


# N=1000000
# beta = np.array([3,1.5,0,0,2,0,0,0])
# with open('sim logistic data.csv', mode='w',newline='') as f:
#     f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     f_writer.writerow(["y","x1","x2","x3","x4","x5","x6","x7","x8"])
#     for n in range(0,N):
#         #np.random.seed(n)
#         x = np.random.normal(0,1,8)
#         p = logistic(x@beta)
#         y = np.random.binomial(1, p, size=None)
#         f_writer.writerow([y]+x.tolist())
# f.close()


# In[ ]:


# simu_data=np.genfromtxt('sim logistic data.csv', delimiter=',')
# y=simu_data[1:,0]
# x=simu_data[1:,1:]
# beta_OLS=minimize(logistic_likelihood, beta, method="Powell",args=(y,x)).x
# beta_adaptive=minimize(adaptive, [3,1.5,0,0,2,0,0,0], method="Powell",args=(logistic_likelihood,y,x,beta_OLS,0.001)).x


# In[ ]:


# beta_OLS


# In[ ]:


# def sim_saver(k_N,m_N):
#     alpha=(k_N*m_N)/N
#     SE_fullsample=np.sqrt((1+1/alpha)*np.linalg.inv(logistic_second_derivative(beta,y,x)[[0,1,4],:][:,[0,1,4]])[[0,1,2],[0,1,2]])
#     with open('k_N='+str(k_N)+'_'+'m_N='+str(m_N)+'_'+'.csv', mode='w',newline='') as f:
#         f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         f_writer.writerow(['beta_1','beta_2','beta_3','beta_4','beta_5','beta_6','beta_7','beta_8',\
#                            'lasso beta_1','lasso beta_2','lasso beta_3','lasso beta_4',\
#                            'lasso beta_5','lasso beta_6','lasso beta_7','lasso beta_8',\
#                           'subsample SE beta_1','subsample SE beta_2','subsample SE beta_5','full sample SE beta_1',\
#                           'full sample SE beta_2','full sample SE beta_5'])
#         for i in range(0,1000):
#             result=subbag('sim logistic data.csv',k_N,m_N,logistic_likelihood)
#             estimate=minimize(LSA, beta+1, method='Powell',args=(result[0],result[1],0)).x
#             estimate_lasso=minimize(LSA, beta, method='Powell',args=(result[0],result[1],0.001)).x
#             SE_subsample=np.sqrt((1+1/alpha)*np.linalg.inv(np.mean(result[1],axis=0)*N)[[0,1,4],[0,1,4]])
#             f_writer.writerow(estimate.tolist()+estimate_lasso.tolist()+SE_subsample.tolist()+SE_fullsample.tolist())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # BIC

# In[812]:





# In[14]:


# print(SBIC('sim linear data.csv',1000,10,mse,lamda_max=1,interval=0.1,scale=False))


# In[ ]:





# In[ ]:


# In[121]:


A= np.random.randn(1000, 50)


# In[129]:


A[np.arange(12), np.arange(12) ]


# In[ ]:





# In[ ]:





# In[ ]:




