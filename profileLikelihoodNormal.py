# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:14:27 2021

@author: 16617

METHODS
``pdf(x, mean=None, cov=1, allow_singular=False)``

Probability density function.

``logpdf(x, mean=None, cov=1, allow_singular=False)``

Log of the probability density function.

``cdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5)``

Cumulative distribution function.

``logcdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5)``

Log of the cumulative distribution function.

``rvs(mean=None, cov=1, size=1, random_state=None)``

Draw random samples from a multivariate normal distribution.

``entropy()``


"""

import numpy as np
from scipy.stats import multivariate_normal, chi2
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def twoLogLike(theta, x, xStar):
    meanStar = xStar[:2]
    covStar = np.array([[xStar[2],xStar[3]],[xStar[3],xStar[4]]])
    log_xStar = multivariate_normal.logpdf(meanStar, cov=covStar).sum()
    
    meanX = theta[:2]
    covX = np.array([[theta[2],theta[3]],[theta[3],theta[4]]])
    log_x = multivariate_normal.logpdf(meanX, cov=covX).sum()
    
    twoLogLike = 2*(log_xStar-log_x)
    
    return twoLogLike
    

# generate a sample 
# true values
meanT = [0.5, -0.2]
covT = [[2.0, 0.3],[0.3,0.5]]


rNv = multivariate_normal(meanT, covT)

x = (rNv.rvs(size=20))


logMLE_x = -multivariate_normal.logpdf(x, meanT, covT).sum()


# now from the data
meanXdata = np.mean(x,axis=0)
covXdata = np.cov(x, rowvar=False)


#This is the true logMLE
logMLEtrue = multivariate_normal.logpdf(x, mean=meanT, cov=covT).sum()
print(logMLEtrue)

#This is the logMLE from the data sample
logMLExStar = multivariate_normal.logpdf(x, mean=meanXdata, cov=covXdata).sum()
print(logMLExStar)

##  frist get the profile likelihood of mean of x only ##########
# that is, keep y, and cov constant, one dimensonal case

meanXdata


chi2Val = chi2.ppf(0.95,1)



# find Profile likelihood CI for some parameter of MVN

def axBdds(meanData, covData, x, index):
        
    lclMin= meanData[index] - 5*covData[index,index]
    lclMax = meanData[index]
    xStar = list(meanData)+list(covData.flatten())
    xStar[4]=xStar[5]
    xStar= xStar[:5]
    
    theta = xStar.copy()
    
    while(True):
        bot = lclMin
        top = lclMax
       
        theta[index] = (bot+top)/2.   # whichever parameter we seek
        ll = twoLogLike(theta, x, xStar)
        
        if ll < chi2Val:
            lclMax = theta[index]
        else:
            lclMin=theta[index]
            
        if (lclMax-lclMin) < 1.e-4: break
    

    axMin = (theta[index]*1.05)
    s = np.sign(axMin)
    axMin = abs(axMin)*s
    
    
    lclMin= meanData[index] 
    lclMax = meanData[index] +  5*covData[index,index]
    xStar = list(meanXdata)+list(covXdata.flatten())
    xStar[4]=xStar[5]
    xStar= xStar[:5]
    
    theta = xStar.copy()
    
    while(True):
        bot = lclMin
        top = lclMax
       
        theta[index] = (bot+top)/2.
        ll = twoLogLike(theta, x, xStar)
        
        if ll > chi2Val:
            lclMax = theta[index]
        else:
            lclMin=theta[index]
            
        if (lclMax-lclMin) < 1.e-4: break
    
    axMax = (1.05*theta[index])
    s = np.sign(axMax)
    axMax = abs(axMax)*s 
    
    return (axMin,axMax)
    

xBdds = axBdds(meanXdata, covXdata, x,0)
xAxis = np.arange(xBdds[0], xBdds[1], 0.05)

# MLE from the data        
xStar = list(meanXdata)+list(covXdata.flatten())
xStar[4]=xStar[5]
xStar= xStar[:5]  

saveX = []
plMeanX = xStar.copy()
for m0 in xAxis:
    plMeanX[0] = m0
    saveX.append(twoLogLike(plMeanX, x, xStar))
  

plt.figure(1)
plt.axhline(y=chi2Val, color='r', linestyle=':')
plt.plot(xAxis, saveX, label='sample Mean {}'.format(np.round(meanXdata[0],2)))
plt.title('X component')
plt.legend()
plt.text(0.0, 5,'true Mean {}'.format(meanT[0]), fontsize='large')
plt.grid()
plt.show()



yBdds = axBdds(meanXdata, covXdata, x, 1)
yAxis = np.arange(yBdds[0], yBdds[1], 0.05)


#xStar = list(meanXdata)+list(covXdata.flatten())
#xStar[4]=xStar[5]
#xStar= xStar[:5]
plMeanY = xStar.copy()

saveY = []
for n0 in yAxis:
    plMeanY[1] = n0
    saveY.append(twoLogLike(plMeanY, x, xStar)) 

plt.figure(2)
plt.axhline(y=chi2Val, color='r', linestyle=':')
plt.plot(yAxis, saveY, label='sample Mean {}'.format(np.round(meanXdata[0],2)))
plt.title('Y component')
plt.legend()
plt.text(-0.6, 5,'true Mean ={}'.format(-0.2), fontsize='large')
plt.grid()
plt.show()

######################################################################
###  now lets get the confidence region for the mean vector (x,y)
## from the parameter PL confidence I's, we know that the bivariate
## ci is within


# bigger region!:
    
###   y bounds   Profile likelihood

xBdds = axBdds(meanXdata,covXdata,x, 0)
yBdds = axBdds(meanXdata,covXdata,x, 1)
saveXbds = list(xBdds).copy()
saveYbds = list(yBdds).copy()

    
yBdds = list(yBdds)
xBdds = list(xBdds)

yBdds[0] = 1.5*yBdds[0]
s = np.sign(yBdds[0])
yBdds[0] = abs(yBdds[0])*s

yBdds[1] = 1.5*yBdds[1]
s = np.sign(yBdds[1])
yBdds[1] = abs(yBdds[1])*s

xBdds[0] = 1.5*xBdds[0]
s = np.sign(xBdds[0])
xBdds[0] = abs(xBdds[0])*s

xBdds[1] = 1.5*xBdds[1]
s = np.sign(xBdds[1])
xBdds[1] = abs(xBdds[1])*s



chi2Val= chi2.ppf(0.95, df=2)  # estimate two parameters
yAxis = np.arange(yBdds[0], yBdds[1], 0.091)
xAxis = np.arange(xBdds[0], xBdds[1], 0.0921)

m = len(xAxis)
n = len(yAxis)
xIndex = []
yIndex = []


for i in range(m):
    for j in range(n):
        
        x0 = np.array([xAxis[i],yAxis[j]])
        
        x0 = list(x0)+list(covXdata.flatten())
        x0[4]=x0[5]
        x0= x0[:5]
        
        logPL2 = twoLogLike(x0, x, xStar)
       
        if logPL2 < chi2Val:
            xIndex.append(i)
            yIndex.append(j)
            
plt.plot(xAxis[xIndex],yAxis[yIndex],'k.')
plt.xlim(xAxis[0],xAxis[m-1])
plt.ylim(yAxis[0],yAxis[n-1])
plt.title('bivariate confidence region for mean vector')
plt.plot(meanXdata[0],meanXdata[1], 'rX', label='sampled')
plt.plot(meanT[0], meanT[1], 'bX', label='truth')
plt.hlines(saveYbds, xmin=xBdds[0], xmax=xBdds[1], color='darksalmon', label='single value CI')
plt.vlines(saveXbds, ymin= xBdds[0], ymax = xBdds[1],color='darksalmon')
plt.legend()

plt.grid()
        
        
        
        
        
        
        
