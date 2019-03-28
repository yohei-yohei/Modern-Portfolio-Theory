#This program is getting three companies stock prices data from yahoo finance
#and using it for simulation of investing money based on certain weights
#Three people invest in different weights and showing the result by a graph.
#Ryan will invest money 100% for the company which had the highest rate.
#Shuba will invest money 33.333% for the each companies.
#Garret will invest money based on modern portfolio theory.
#Garret will take same expected return as Ryan's one.
#@Author:Yohei Sato

import pandas as pd
import pandas_datareader.data as web 
import datetime
import matplotlib.pyplot as plt   
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches2
import matplotlib.patches as mpatches3
import pylab as pyla
import math
import numpy as np
import itertools
from numpy.linalg import inv

#Getting variance by using the formula
def variancecal(i, j, l, varfirst, varsecond, varthird, \
                coonetwo, coonethree, cotwothree):
    return   ((i/100)**2)*(varfirst**2) + ((j/100)**2)*(varsecond**2) + \
           ((l/100)**2)*(varthird**2) + \
           2*(i/100)*(j/100)*varfirst*varsecond*coonetwo + \
           2*(i/100)*(l/100)*varfirst*varthird*coonethree + \
           2*(j/100)*(l/100)*varsecond*varthird*cotwothree
#Getting the expected return by using the formula
def expectedreturn(expect1, expect2, expect3, weight1, weight2, weight3):
    return expect1*weight1 + expect2*weight2 + expect3*weight3

#Getting sum and the list based on Adjested close price from the stock data 
def getSumandList(data):
    theList = []
    i = 0
    theSum = 0;
    while(i<len(data1) -1):
        temp = (data.at[data.index[i], 'Adj Close'])
        i+=1
        temp2 = math.log((data.at[data.index[i], 'Adj Close'])/temp)
        theList.append(temp2)
        theSum += temp2
    return theSum, theList

#Getting variance from each yahoo finance adjested close price and expected return
def getVariance(length, theList, er):
    i = 0
    preSS = 0
    SS = 0
    while i < length -1:
        valueMinusMean = math.exp(theList[i]) - er
        preSS = valueMinusMean**2
        SS += preSS
        i +=1
    return SS/(length-2)

#Getting covariance from each yahoo finance adjested close price and expected return
def getCoVariance(length, theFirstList, theSecondList, er1, er2):
    i = 0
    preSSCO = 0
    SSCO = 0
    while i < length -1:
        valueMinusMean1 = math.exp(theFirstList[i]) - er1
        valueMinusMean2 = math.exp(theSecondList[i]) - er2
        preSSCO = valueMinusMean1*valueMinusMean2
        SSCO += preSSCO
        i +=1
    return SSCO/(length-2)



#Input about companies names which we wanna use for stock data
data1name = input("Input the first company stock name:")
data2name = input("Input the second company stock name:")
data3name = input("Input the third company stock name:")
#EP = float(input("Input expected return portfolio"))


# We look at prices over the past year, start at 01/01/17 for fiding weight by MPT
start = datetime.datetime(2017, 1, 1)
end = datetime.datetime(2017, 12, 31)
data1 = web.DataReader(data1name, "yahoo", start, end)
data2 = web.DataReader(data2name, "yahoo", start, end)
data3 = web.DataReader(data3name, "yahoo", start, end)
 

raw_input = ""



#Getting Prices over the past year for simulation how much they can gain and lose.
newstart = datetime.datetime(2018, 1, 1)
newend = datetime.datetime(2018, 12, 31)
newdata1 = web.DataReader(data1name, "yahoo", newstart, newend)
newdata2 = web.DataReader(data2name, "yahoo", newstart, newend)
newdata3 = web.DataReader(data3name, "yahoo", newstart, newend)


pyla.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots
pd.core.frame.DataFrame



sum1 = 0
sum2 = 0
sum3 = 0

list1 = []
list2 = []
list3 = []

sum1, list1 = getSumandList(data1)
sum2, list2 = getSumandList(data2)
sum3, list3 = getSumandList(data3)



prevariance1 = 0;
prevariance2 = 0;
prevariance3 = 0;
SS1 = 0;
SS2 = 0;
SS3 = 0;
SSCO11 = 0;
SSCO12 = 0;
SSCO13 = 0;
SSCO23 = 0;



preexpectedreturn1 = sum1
preexpectedreturn2 = sum2
preexpectedreturn3 = sum3


#Company1's expected return
expectedreturn1 = math.exp(preexpectedreturn1)
#Company2's expected return
expectedreturn2 = math.exp(preexpectedreturn2)
#Company3's expected return
expectedreturn3 = math.exp(preexpectedreturn3)

averageExpectedreturn = \
                      (expectedreturn1+expectedreturn2+expectedreturn3)/3

print("company1 expected return",expectedreturn1)
print("company2 expected return",expectedreturn2)
print("company3 expected return",expectedreturn3)

print("length data1", len(data1))
print("length data2", len(data2))
print("length data3", len(data3))

variance1 = getVariance(len(data1), list1, expectedreturn1)
variance2 = getVariance(len(data2), list2, expectedreturn2)
variance3 = getVariance(len(data3), list3, expectedreturn3)


covariance12 = getCoVariance(len(data1),\
                             list1,list2,expectedreturn1,expectedreturn2)
covariance13 = getCoVariance(len(data1),\
                             list1,list3,expectedreturn1,expectedreturn3)
covariance23 = getCoVariance(len(data2),\
                             list2,list3,expectedreturn2,expectedreturn3)

omega = np.array( ((variance1, covariance12, covariance13), \
                   (covariance12, variance2, covariance23), \
                   (covariance13, covariance23, variance3)))
omegainverse = inv(omega)
expectedreturnmatrix = \
        np.array( ((expectedreturn1),(expectedreturn2),(expectedreturn3)))
newomega = np.dot(omega,omegainverse)
verticalmatrix111 = np.array( ((1),(1),(1)))
holizontalmatrix111 = np.array( ((1,1,1 )))




temp = 100;




ryanweight1 = 0
ryanweight2 = 0
ryanweight3 = 0

company1money = (newdata1.at[newdata1.index[0], 'Adj Close'])
company2money = (newdata2.at[newdata1.index[0], 'Adj Close'])
company3money = (newdata3.at[newdata1.index[0], 'Adj Close'])

print("company1money is ", company1money)
print("company2money is ", company2money)
print("company3money is ", company3money)


GarretMoneyData = []
ShubhaMoneyData = []

ryanstocknumber = 0
RyanMoneyData = []

#Ryan invests every money for the company which has the highest growth rate.
if(expectedreturn1 >expectedreturn2  and expectedreturn1  > expectedreturn3):
    ryanweight1 = 1
    ryanstocknumber = 5000//company1money
    RyansRemainder = 5000 - (ryanstocknumber*company1money)
    i = 0
    while i < len(newdata1) -1:
            temp = (newdata1.at[newdata1.index[i], 'Adj Close'])
            i +=1
            RyanTempMoney = temp*ryanstocknumber+RyansRemainder
            RyanMoneyData.append(RyanTempMoney)
            
elif(expectedreturn2  > expectedreturn3 and expectedreturn2 > expectedreturn1 ):
    ryanweight2 = 1
    ryanstocknumber = 5000/company2money
    i = 0
    RyansRemainder = 5000 - (ryanstocknumber*company2money)
    while i < len(newdata2) -1:
            temp = (newdata2.at[newdata2.index[i], 'Adj Close'])
            i +=1
            RyanTempMoney = temp*ryanstocknumber+RyansRemainder
            RyanMoneyData.append(RyanTempMoney)
           
else:
    ryanweight3 = 1
    ryanstocknumber = 5000/company3money
    i = 0
    RyansRemainder = 5000 - (ryanstocknumber*company3money)
    while i < len(newdata3) -1:
            temp = (newdata3.at[newdata3.index[i], 'Adj Close'])
            i +=1
            RyanTempMoney = temp*ryanstocknumber+RyansRemainder
            RyanMoneyData.append(RyanTempMoney)
            
RyanExpectedreturn = expectedreturn(expectedreturn1,expectedreturn2,\
                                    expectedreturn3,ryanweight1,ryanweight2,ryanweight3)  


#EP for getting Garret's weight
EP = RyanExpectedreturn

print("expectedreturnmatrix is", expectedreturnmatrix)


A = (holizontalmatrix111.dot(omegainverse)).dot(expectedreturnmatrix)
B = (expectedreturnmatrix.dot(omegainverse)).dot(expectedreturnmatrix)
C = (holizontalmatrix111.dot(omegainverse)).dot(verticalmatrix111)

lamda1 = 2*((C*EP-A)/(B*C-(A**2)))
lamda2 = 2*((B-A*EP)/(B*C-(A**2)))

optimizedWeightArray = omegainverse.dot((((lamda1/2)*\
                        (expectedreturnmatrix))+((lamda2/2)*(verticalmatrix111))))
variance = 0

optimizedWeight1 = optimizedWeightArray[0]
optimizedWeight2 = optimizedWeightArray[1]
optimizedWeight3 = optimizedWeightArray[2]


print("Ryans'Remainder", RyansRemainder)

GarretExpectedreturn = expectedreturn(expectedreturn1,expectedreturn2,\
        expectedreturn3,optimizedWeight1,optimizedWeight2,optimizedWeight3)

ShubhaExpectedreturn = expectedreturn(expectedreturn1,\
                            expectedreturn2,expectedreturn3,0.4,0.3,0.3)

Garretnumberofstock1 = (optimizedWeight1*5000)//company1money
Garretnumberofstock2 = (optimizedWeight2*5000)//company2money
Garretnumberofstock3 = (optimizedWeight3*5000)//company3money
Shubhanumberofstock1 = ((1/3)*5000)//company1money
Shubhanumberofstock2 = ((1/3)*5000)//company2money
Shubhanumberofstock3 = ((1/3)*5000)//company3money


GarretsRemainder = 5000 - (company1money*Garretnumberofstock1+company2money* \
                           Garretnumberofstock2+company3money*Garretnumberofstock3)
ShubhasRemainder = 5000 - (company1money*Shubhanumberofstock1+company2money* \
                           Shubhanumberofstock2+company3money*Shubhanumberofstock3)

print("GarretsRemainder", GarretsRemainder)
print("ShubhasRemainder", ShubhasRemainder)


i = 0
#Putting everyday Garret/Shubha's money for each their list
while i < len(newdata1) -1:
            temp1 = (newdata1.at[newdata1.index[i], 'Adj Close'])
            temp2 = (newdata2.at[newdata2.index[i], 'Adj Close'])
            temp3 = (newdata3.at[newdata3.index[i], 'Adj Close'])
            i +=1
            GarretTempMoney=temp1*Garretnumberofstock1+temp2* \
                             Garretnumberofstock2+temp3*\
                        Garretnumberofstock3+GarretsRemainder
            ShubhaTempMoney=temp1*Shubhanumberofstock1+temp2* \
                             Shubhanumberofstock2+temp3*\
                        Shubhanumberofstock3+ShubhasRemainder
            GarretMoneyData.append(GarretTempMoney)
            ShubhaMoneyData.append(ShubhaTempMoney)



print("Garrett invested" ,round(100*optimizedWeight1, 3), "% weight for", data1name)
print("Garrett invested" ,round(100*optimizedWeight2, 3), "% weight for", data2name)
print("Garrett invested" ,round(100*optimizedWeight3, 3), "% weight for", data3name)
print("Garrett's expected return is ", GarretExpectedreturn)

print("Shuba invested" ,round(100*(1/3),4), "% weight for", data1name)
print("Shuba invested" ,round(100*(1/3),4), "% weight for", data2name)
print("Shuba invested" ,round(100*(1/3),4), "% weight for", data3name)
print("Shubha's expected return is ", ShubhaExpectedreturn)

print("Ryan invested" ,100*ryanweight1, "% weight for", data1name)
print("Ryan invested" ,100*ryanweight2, "% weight for", data2name)
print("Ryan invested" ,100*ryanweight3, "% weight for", data3name)
print("Ryan's expected return is ", RyanExpectedreturn)

#Putting their remainders back into the final result
#GarretMoneyData.append(GarretMoneyData[-1]+GarretsRemainder)
#ShubhaMoneyData.append(ShubhaMoneyData[-1]+ShubhasRemainder)
#RyanMoneyData.append(RyanMoneyData[-1]+RyansRemainder)


#Showing the graph
plt.plot(GarretMoneyData, color='green', label="Garrett")
plt.plot(RyanMoneyData, color='blue', label="Ryan")
plt.plot(ShubhaMoneyData, color='red', label="Shubha")
plt.legend(loc='best')
plt.show()
        

