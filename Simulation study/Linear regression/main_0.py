rep_ind_current=1
#!/usr/bin/env python
# coding: utf-8

# In[171]:


# rep_ind_current = 1
#!/usr/bin/env python
# coding: utf-8

# The following are the commonly used packages

#import clubear as cb
import numpy as np
import csv
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import time




# In[172]:


# The code in this cell is exactly from clubear


import os
import random 
import numpy as np
import multiprocessing
import pandas as pd





# In[173]:


# define some loss functions
def mse(beta,y,x):
    return sum((y-x@beta)**2)/len(y)
    
def adaptive(beta,f,y,x,beta_0,lamda):
    return f(beta,y,x)+lamda*sum(abs(beta)/abs(beta_0))
    
def mse_first_derivative(beta,y,x):
    return -2*(x.T@(y-x@beta))/len(y)
    
def mse_second_derivative(beta,y,x):
    return 2*x.T@x/len(y)

# def var_of_mse_grad_linear(x, y, beta):
#     n, p =x.shape
#     r = y - x @ beta                         # residuals, (n,)
#     g_i = -2 * x * r[:, None]/len(y)              # (n,p), each row is g_i(beta)

#     gbar = g_i.mean(axis=0)                  # (p,)
#     centered = g_i - gbar[None, :]           # (n,p)

#     cov_gi = centered.T @ centered / (n - 1) # (p,p) sample covariance of g_i
#     return cov_gi



def logistic(z):
    return np.exp(z) / (1 + np.exp(z))
    
def logistic_likelihood(beta,y,x):
    X=x
    Y=y
    p=logistic(X@beta)
    return -sum((Y*np.log(p)+(1-Y)*np.log(1-p)))/len(y)

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







# In[174]:


# This function is for a subsample's estimation

def subsample_estimate(file_name,subsize,f): # subsize is k_N; f is the loss

    # extract one subsample
    simu_data = pd.read_csv(file_name, header = 0).to_numpy()
    subsample = simu_data[random.sample(range(1,len(simu_data)), k = subsize)]
    y_subsample = subsample[:,0]
    x_subsample = subsample[:,1:]
    beta_true = np.array([3,1.5,0,0,2,0,0,0])

    # obtain subsample estimates
    beta_subsample = minimize(f, beta_true, method='BFGS',    
                            args = (y_subsample,x_subsample)).x
    # obtain captial Sigma for linear regssion
    second_derivative_subsample = mse_second_derivative(beta_subsample, y_subsample, x_subsample)

    # obtain the middle matrix in the sandwich covariance matrix (using true beta or estimate)
    # first_derivative_subsample = mse_first_derivative(beta_subsample, y_subsample, x_subsample)
    # first_derivative_true =  mse_first_derivative(beta_true, y_subsample, x_subsample) 
    Sigma_hat_variance_subsample = np.sum( (y_subsample - x_subsample @ beta_subsample)**2 )  / (len(y_subsample))
    Sigma_hat_variance_true = np.sum( (y_subsample - x_subsample @ beta_true)**2 )  / (len(y_subsample))

    return beta_subsample, second_derivative_subsample, Sigma_hat_variance_subsample, Sigma_hat_variance_true





# In[175]:


# define the least square approxatimation
def LSA(beta,beta_subsample,second_derivative_subsample,lamda=0):
    approx=0
    m_N=len(beta_subsample)
    for i in range(0,m_N):
        #iterate through m_N subsamples
        approx += (beta-beta_subsample[i]).transpose()@second_derivative_subsample[i]@(beta-beta_subsample[i])
    approx = approx/m_N
    weights = np.mean(beta_subsample, axis=0)
    weights[:]*= weights
    return approx + lamda*sum(abs(beta)/np.abs(weights)**1)


# In[176]:


# This function collect the information from each subsample in a list

def subbag(file_name,k_N,m_N,f,N):
    # First generate full sample
    beta = np.array([3,1.5,0,0,2,0,0,0])
    with open(file_name, mode='w',newline='') as file:
        f_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(["y","x1","x2","x3","x4","x5","x6","x7","x8"])
        for n in range(0,N):
            x = np.random.normal(0,1,8)
            y = x@beta + np.random.normal(0,1,1)
            f_writer.writerow(y.tolist()+x.tolist())
    file.close()

    # create lists to collect the information
    beta_subsample = []
    second_derivative_subsample =[]
    Sigma_hat_variance_subsample = []
    Sigma_hat_variance_true = []
    for i in range(0,m_N):
        # the result from the above function
        result = subsample_estimate(file_name = file_name, subsize = k_N, f=f)
        beta_subsample += [result[0]]
        second_derivative_subsample += [result[1]]
        Sigma_hat_variance_subsample  += [result[2]]
        Sigma_hat_variance_true += [result[3]]
    # Delete the file
    os.remove(file_name)
    return beta_subsample, second_derivative_subsample, Sigma_hat_variance_subsample, Sigma_hat_variance_true



# In[177]:


# define the function that selects the best lambda
def SBIC(k_N, m_N, result, lamda_constant = 1, interval = 0.000001, scale = True):
    BIC_min = float('inf')
    beta_true = np.array([3,1.5,0,0,2,0,0,0])
    for log_scale in range(0, int(-np.log(interval))):
        lamda = lamda_constant * 10 ** (-log_scale)
        alpha = (k_N * m_N)/N
        estimate = minimize(LSA, beta_true, method = "Powell", args = (result[0], result[1], lamda)).x
        df = sum(estimate!=0)
        if scale == True:
            BIC = k_N * LSA(estimate, result[0], result[1], lamda = lamda) + df * np.log(N)
        if scale == False:
            BIC = LSA(estimate, result[0], result[1],lamda = lamda) + df * np.log(N)
        if BIC < BIC_min:
            BIC_min = BIC
            lamda_min = lamda
            estimate_optimal = estimate
    return BIC_min, lamda_min, estimate_optimal


# In[178]:


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



beta_true = np.array([3,1.5,0,0,2,0,0,0])






# In[179]:


def sim_saver(k_N,m_N,N):
    alpha=(k_N * m_N)/N
    # prepare writing for subsample results
    # SE_fullsample=np.sqrt((1+1/alpha)*np.linalg.inv(second_derivative(mse,beta,y,x)[[0,1,4],:][:,[0,1,4]])[[0,1,2],[0,1,2]])
    # If summary file not exist, create a new one
    file_name = '../result/N=' + str(N) + '_k_N='+str(k_N)+'_'+'m_N='+str(m_N)+'_'+'linear_reg.csv'
    if (not (os.path.exists(file_name))):
        with open(file_name, mode='w',newline='') as f:
            f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            f_writer.writerow(['index','beta_1','beta_2','beta_3','beta_4','beta_5','beta_6','beta_7','beta_8',\
                               'lasso beta_1','lasso beta_2','lasso beta_3','lasso beta_4',\
                               'lasso beta_5','lasso beta_6','lasso beta_7','lasso beta_8',\
                              'subsample SE1 beta_1','subsample SE1 beta_2','subsample SE1 beta_5',
                              'subsample SE2 beta_1','subsample SE2 beta_2','subsample SE2 beta_5',
                              'subsample SE3 beta_1','subsample SE3 beta_2','subsample SE3 beta_5',
                              'subsample SE4 beta_1','subsample SE4 beta_2','subsample SE4 beta_5',
                              'subsample SE5 beta_1','subsample SE5 beta_2','subsample SE5 beta_5',
                              #  'full sample SE beta_1',\
                              # 'full sample SE beta_2','full sample SE beta_5',
                              'CI beta_1', "CI beta_2", "CI beta_5",
                              'CI2 beta_1', "CI2 beta_2", "CI2 beta_5",
                              'CI3 beta_1', "CI3 beta_2", "CI3 beta_5",
                              'CI4 beta_1', "CI4 beta_2", "CI4 beta_5",
                              'CI5 beta_1', "CI5 beta_2", "CI5 beta_5",
                              # 'CI6 beta_1', "CI6 beta_2", "CI6 beta_5", 
                              # 'CI7 beta_1', "CI7 beta_2", "CI7 beta_5",
                               "BIC_min", "lamda_min",
                              "time", "memory"])
    # simulation start writing into the corresponding files            
    with open(file_name, mode = 'a',newline = '') as f:
        
        f_writer = csv.writer(f, delimiter = ',', quotechar = '"', quoting=csv.QUOTE_MINIMAL)
        
        for i in range(0,2):
            
            random.seed(rep_ind_current+i)
            
            start_time = time.time()
            
            # obtain the collection from subbag files
            result = subbag('sim linear data_N=' + str(N) + '_' + str(rep_ind_current+i) + '.csv',k_N,m_N,mse,N)
            # Simple average of subbagging estimates
            estimate = np.mean(result[0], axis = 0)
            # LSA minmizer; we set lambda small to avoid potential bias for now
            # The optimizer method Powerll can give exactly value of 0 when intial value is true beta
            # estimate_lasso = minimize(LSA, beta_true, method='Powell',args=(result[0],result[1],0.000001)).x
            lasso_result = SBIC(k_N, m_N, result)
            estimate_lasso = lasso_result[2]
            BIC_min = lasso_result[0]
            lamda_min = lasso_result[1]

            # First kind of SE calculation; i.e., based on bootstrapping
            SE1_subsample = np.sqrt(k_N * (1 + 1/alpha) * ((np.array(result[0]) - estimate).T@(np.array(result[0]) - estimate))[[0,1,4],[0,1,4]]/m_N/N)
            end_time = time.time()

            # Other sandwitch matrix SE calculation
            SE2_subsample = np.sqrt((1 + 1/alpha)/N * np.mean(result[2]) * np.identity(8)[[0,1,4],[0,1,4]])
            SE3_subsample = np.sqrt((1 + 1/alpha)/N * np.mean(result[2]) * np.identity(8)[np.ix_([0,1,4],[0,1,4])])[[0,1,2],[0,1,2]]
            SE4_subsample = np.sqrt((1 + 1/alpha)/N * np.mean(result[3]) * np.identity(8)[[0,1,4],[0,1,4]])
            SE5_subsample = np.sqrt((1 + 1/alpha)/N * np.mean(result[3]) * np.identity(8)[np.ix_([0,1,4],[0,1,4])])[[0,1,2],[0,1,2]]
        
            # Coverage of confidence interval based on the SE
            CI1_subsample = (estimate_lasso[[0,1,4]] + norm.ppf(0.975) * SE1_subsample > [3, 1.5, 2]) * (estimate_lasso[[0,1,4]] - norm.ppf(0.975) * SE1_subsample < [3, 1.5, 2])
            CI2_subsample = (estimate_lasso[[0,1,4]] + norm.ppf(0.975) * SE2_subsample > [3, 1.5, 2]) * (estimate_lasso[[0,1,4]] - norm.ppf(0.975) * SE2_subsample < [3, 1.5, 2])
            CI3_subsample = (estimate_lasso[[0,1,4]] + norm.ppf(0.975) * SE3_subsample > [3, 1.5, 2]) * (estimate_lasso[[0,1,4]] - norm.ppf(0.975) * SE3_subsample < [3, 1.5, 2])            
            CI4_subsample = (estimate_lasso[[0,1,4]] + norm.ppf(0.975) * SE4_subsample > [3, 1.5, 2]) * (estimate_lasso[[0,1,4]] - norm.ppf(0.975) * SE4_subsample < [3, 1.5, 2])            
            CI5_subsample = (estimate_lasso[[0,1,4]] + norm.ppf(0.975) * SE5_subsample > [3, 1.5, 2]) * (estimate_lasso[[0,1,4]] - norm.ppf(0.975) * SE5_subsample < [3, 1.5, 2])            
            # CI6_subsample = (estimate[[0,1,4]] + 1.96 * SE1_subsample > [3, 1.5, 2]) * (estimate[[0,1,4]] - 1.96 * SE1_subsample < [3, 1.5, 2])            
            # CI7_subsample = (estimate[[0,1,4]] + 1.96 * SE2_subsample > [3, 1.5, 2]) * (estimate[[0,1,4]] - 1.96 * SE2_subsample < [3, 1.5, 2])            

            
            f_writer.writerow(([rep_ind_current+i]) + 
                              estimate.tolist() + 
                              estimate_lasso.tolist() +
                              SE1_subsample.tolist() +
                              SE2_subsample.tolist() + 
                              SE3_subsample.tolist() +
                              SE4_subsample.tolist() +
                              SE5_subsample.tolist() +
                              CI1_subsample.astype(int).tolist() +
                              CI2_subsample.astype(int).tolist() +
                              CI3_subsample.astype(int).tolist() +
                              CI4_subsample.astype(int).tolist() +
                              CI5_subsample.astype(int).tolist() +
                              #CI6_subsample.astype(int).tolist() +
                              #CI7_subsample.astype(int).tolist() +
                              ([BIC_min]) +
                              ([lamda_min]) +
                              [end_time-start_time])





# In[180]:


# Settings where m_N are doubled
N = 500000
alpha = 0.2
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)




alpha = 1
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)





alpha = 2
sim_saver(k_N=int(N**(1/4+1/2)),m_N=(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
sim_saver(k_N=int(N**(1/3+1/2)),m_N=(int(alpha * N/(N**(1/3+1/2)))+1), N = N)



# alpha = 2
# sim_saver(k_N=int(N**(1/4+1/2)),m_N=2*(int(alpha * N/(N**(1/4+1/2)))+1), N = N)
# sim_saver(k_N=int(N**(1/3+1/2)),m_N=2*(int(alpha * N/(N**(1/3+1/2)))+1), N = N)





# In[ ]:


# Settings where k_N are halfed
# N = 500000
# alpha = 0.1
# sim_saver(k_N=int(N**(1/4+1/2)/2),m_N=int(alpha * N/(N**(1/4+1/2)/2))+1, N = N)
# sim_saver(k_N=int(N**(1/3+1/2)/2),m_N=int(alpha * N/(N**(1/3+1/2)/2))+1, N = N)

# # In[23]:


# alpha = 0.5
# sim_saver(k_N=int(N**(1/4+1/2)/2),m_N=int(alpha * N/(N**(1/4+1/2)/2))+1, N = N)
# sim_saver(k_N=int(N**(1/3+1/2)/2),m_N=int(alpha * N/(N**(1/3+1/2)/2))+1, N = N)


# # In[24]:


# alpha = 1
# sim_saver(k_N=int(N**(1/4+1/2)/2),m_N=int(alpha * N/(N**(1/4+1/2)/2))+1, N = N)
# sim_saver(k_N=int(N**(1/3+1/2)/2),m_N=int(alpha * N/(N**(1/3+1/2)/2))+1, N = N)








# # In[ ]:


# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:





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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




