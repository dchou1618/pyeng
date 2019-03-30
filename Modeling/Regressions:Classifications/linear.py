#!/usr/bin/env python3

# dependencies

import numpy as np, pandas as pd, math, string, random, sys, time
import seaborn as sns
from copy import *
from matplotlib.pyplot import *
from math import *
from numpy import *

# csv example from https://www.kaggle.com/lava18/google-play-store-apps
###########################################################################
'''Simple Linear Regression'''
###########################################################################

class LinearRegression:
    """Base Class - implementation of multivariate optimization of
    square errors to determine optimal parameters with lowest error (cost)"""
    def __init__(self,data=[],x=[],y=[]):
        self.data = data
        self.x,self.y = x,y
    def getAugmented(self,*args):
        """
        Creates object variable self.augmented that is derived from the
        4 matrix entries calculated from the data in self.augmentMatrix()
        """
        try:
            rows,cols = 2,2
            augmented = [[0]*cols for index in range(rows)]
            matrixTerms = [round(args[0],3),round(args[1],3),
                           round(args[2],3),round(args[3],3)]
            ith = 0
            for i in range(rows):
                for j in range(cols):
                    augmented[i][j] = matrixTerms[ith]
                    ith += 1
            self.augmented = augmented
            return augmented
        except Exception as e:
            print(e, "or not enough arguments")

    def augmentMatrix(self):
        """
        Either evaluate 4 entries of the matrix from a csv dataset
        or use the manufactured x,y data
        """
        if type(self.data) == pd.core.frame.DataFrame:
            a_00,a_01,a_10,a_11 = 0,0,0,0
            for index,row in self.data.iterrows():
                if not math.isnan(row["Rating"]):
                    a_00 += (row["Rating"])**2
                    # sum x_i^2
                    a_01 += row["Rating"]; a_10 += row["Rating"]
                    # sum x_i
                    a_11 += 1
                    # sum 1
            return self.getAugmented(a_00,a_01,a_10,a_11)
        else:
            a_00,a_01,a_10,a_11 = 0,0,0,0
            for i in range(len(self.x)):
                a_00 += self.x[i]**2
                a_01 += self.x[i]; a_10 += self.x[i]
                a_11 += 1
            return self.getAugmented(a_00,a_01,a_10,a_11)

    def cleanReview(self,review):
        """
        Cleans abbreviated reviews (M stands for million)
        """
        if "M" in review:
            review = review.replace("M","")
            review = str(float(review)*10**6)
            return review
        else:
            return review

    def getB(self):
        """
        finds the b vector in the matrix equation Ax = b
        """
        def valid(review):
            """
            if review is only numbers or only digits with M (million)
            """
            return review.isdigit() or (review.replace("M","").isdigit())
        if type(self.data) == pd.core.frame.DataFrame:
            b_00,b_10 = 0,0
            for index,row in self.data.iterrows():
                if not math.isnan(row["Rating"]) and valid(row["Reviews"]):
                    modReviews = float(self.cleanReview(row["Reviews"]))
                    b_00 += row["Rating"]*modReviews
                    b_10 += modReviews
            self.B = [b_00,b_10]
            return self.B
        else:
            b_00,b_10 = 0,0
            for i in range(len(self.x)):
                b_00 += self.x[i]*self.y[i]
                b_10 += self.y[i]
            self.B = [b_00,b_10]
            return self.B

    def basicInverse(self,m):
        # inverse of a matrix
        det = m[0][0]*m[1][1]-m[0][1]*m[1][0]
        m[0][1] *= -1
        m[1][0] *= -1
        m[1][1],m[0][0] = m[0][0],m[1][1]
        for i in range(len(m)):
            for j in range(len(m[0])):
                m[i][j] /= det
        return m

    def mult(self,A,B):
        # multiplies matrices A,B
        def dotProduct(l1,l2):
            result = 0
            for i in range(len(l1)):
                result += (l1[i]*l2[i])
            return result
        return [dotProduct(A[0],B),dotProduct(A[1],B)]

    def getab(self):
        # obtains the parameters by multiplying inverse of
        # augmented matrix with b vector
        parameters = self.mult(self.basicInverse(self.augmented),self.B)
        return parameters

    def displayRegression(self):
        '''
        displays the graph of the linear regression with the plotted points
        '''
        def ridOfNaN(L):
            # removes any "not a number" values
            for i in range(len(L)-1,-1,-1):
                if str(L[i]) == "nan":
                    L.pop(i)
            return L
        if type(self.data) == pd.core.frame.DataFrame:
            self.augmentMatrix(); self.getB()
            a,b = self.getab()
            reviews = [float(self.cleanReview(elem)) \
                       for elem in list(self.data["Reviews"])]
            sns.scatterplot(x=self.data["Rating"],
                 y=reviews)
            x = list(self.data["Rating"])
            y = []
            for elem in x:
                y.append(a*elem+b)
            plot(x,y,"darkgreen")

            title(u"SLR of Data\nY={}x+({})".format(str(a),str(b)))
            ylabel("Reviews")
            xlabel("Ratings")
            show()
        else:
            self.augmentMatrix(); self.getB()
            a,b = self.getab()
            sns.scatterplot(self.x,self.y)
            y = []
            for elem in self.x:
                y.append(a*elem+b)
            plot(self.x,y,"darkblue")
            title(u"SLR of Manufactured Data\nY={}x+({})".format(str(a),str(b)))
            ylabel("Y values")
            xlabel("X values")
            show()

def timerFn(f):
    def g(*args):
        start = time.time()
        runFunc = f(*args)
        end = time.time()
        print("Program ran in {} seconds".format(end-start))
    return g

@timerFn
def runRegression():
    # displays the graph of plotted points and regression
    try:
        # assuming csv file read in
        data = pd.read_csv(sys.argv[1])
        myRegression = LinearRegression(data)
        myRegression.displayRegression()
    except Exception as e:
        # otherwise, generate random data points to regress
        numX = random.randint(50,200)
        numY = numX
        x = [random.uniform(0,10000) for i in range(numX)]
        y = [random.uniform(0,10000) for j in range(numY)]
        myRegression = LinearRegression([],x,y)
        myRegression.displayRegression()

if __name__ == "__main__":
    runRegression()
