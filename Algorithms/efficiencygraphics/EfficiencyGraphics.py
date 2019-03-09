#!/usr/bin/python

import time, random, sys, numpy as np, matplotlib.pyplot as plt

# changes format from e to 10^n
def parsePower(s):
    index = s.find("e")
    try:
        return str(s[:index]) + "*10^%d"%int(s[index+1:])
    except:
        return ""

xData = np.array([])
yData = np.array([])

try:
    for line in sys.stdin:
        line = line.split()
        inputSize,runTime = line[0],line[1]
        if "e" in runTime:
            index = runTime.find("e")
            runTime = float(runTime[:index])*10**int(runTime[index+1:])
        xData = np.append(xData,np.array([int(inputSize)]))
        yData = np.append(yData,np.array([float(runTime)]))
except:
    pass

plt.figure(figsize=(6, 4))
# fitting data points to polynomial regression
polynomialFit = int(sys.argv[1])
z = np.polyfit(xData, yData, polynomialFit)
p = np.poly1d(z)

# formatting display of polynomial
polynomial = ""
pLst = list(p)
for term in range(len(pLst)):
    if len(pLst)-term-1 > 1:
        polynomial += parsePower(str(pLst[term]))+"(x^%d)"%(len(pLst)-term-1)+"+"
    elif len(pLst)-term-1 == 1:
        polynomial += parsePower(str(pLst[term]))+"(x)+"
    else:
        polynomial += parsePower(str(pLst[term]))
print(polynomial)

# plots data
xp = np.linspace(xData.min(), xData.max(), 100)
plt.plot(xData, yData, '.', xp, p(xp), '-')

# display graph
plt.title("Graph For Big-O of Function")
plt.xlabel("Input Sizes (N)")
plt.ylabel("Running Times (Secs)")
plt.show()
