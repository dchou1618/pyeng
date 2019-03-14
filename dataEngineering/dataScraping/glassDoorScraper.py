#!/usr/bin/python

import string, time, random, warnings
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd
from bs4 import BeautifulSoup
from time import *
from pandas import *
from random import *
from requests import *

glassDoor = "https://www.glassdoor.com/Job/pittsburgh-jobs-"+\
            "SRCH_IL.0,10_IC1152990.htm?radius=50"
headers = {'user-agent': 'Mozilla/5.0'}

# Looking into the total number of pages
pages = [""]+["_IP{}".format(val) for val in range(2,21)]

class GlassDoorScraper(object):
    def __init__(self,url=glassDoor):
        self.url = url
    def getRatingStars(self,containerText):
        ratingStars = containerText[:3]
        try:
            float(ratingStars)
            containerText = containerText.replace(ratingStars,"")
            return ratingStars, containerText
        except Exception as e:
            logoIndex = containerText.find(" ")
            containerText = containerText[logoIndex:]
            return "NaN", containerText
    def getLocPosition(self,containerText):
        paIndex = containerText.find("PA")
        cityIndex = paIndex
        while paIndex > 0:
            cityIndex -= 1
            if containerText[cityIndex].isupper():
                break
        location = containerText[cityIndex:paIndex+2]
        role = containerText[:cityIndex]
        role = role.replace("–","").strip()
        containerText = containerText[paIndex+2:]
        return location, role, containerText
    def getSalary(self,containerText):
        dollarIndex = containerText.find("$")
        try:
            assert(dollarIndex != -1)
            result = ""
            numDollars = containerText.count("$")
            if numDollars > 1:
                for i in range(numDollars):
                    salary = containerText[dollarIndex:dollarIndex+3]
                    result += salary+"-"
                    containerText = containerText.replace(salary,"")
                    dollarIndex = containerText.find("$")
                return result[:-1], containerText
            else:
                salary = containerText[dollarIndex:dollarIndex+3]
                containerText = containerText.replace(salary,"")
                return salary, containerText
        except:
            return "NaN",containerText
    def cleanContainer(self,container):
        jobText = container.text.strip()
        ratingStars, jobText = self.getRatingStars(jobText)
        salary, jobText = self.getSalary(jobText)
        location, role, jobText = self.getLocPosition(jobText)
        return ratingStars, salary, role, location
    def scrape(self):
        ratings,roles,salaries,locations=[],[],[],[]
        pageRequests = 0
        # Begin recording time for web scraping
        start = time()
        for page in pages:
            glassDoor = "https://www.glassdoor.com/Job/pittsburgh-jobs-"+\
                        "SRCH_IL.0,10_IC1152990{}.htm?radius=50".format(page)
            response = get(glassDoor,headers = headers)
            # increment pageRequest
            pageRequests += 1
            timeElapsed = time()-start
            print("Page Requests: %d, Elapsed Time: %d seconds"
                  %(pageRequests,timeElapsed))
            if response.status_code != 200:
                warn("Bad status code %d"%(response.status_code))
            htmlBSoup = BeautifulSoup(response.text, "html.parser")
            glassDoorContainers = htmlBSoup.find_all('li',class_="jl")
            for i in range(len(glassDoorContainers)):
                container = glassDoorContainers[i]
                rating,salary,job,location = self.cleanContainer(container)
                ratings.append(rating);salaries.append(salary)
                roles.append(job);locations.append(location)
        glassDoorData = pd.DataFrame({"Ratings":ratings,"Jobs":roles,
                                      "Salaries":salaries,
                                      "Locations":locations})
        glassDoorData.to_csv("glassDoorData.csv")

def glassDoorDiagnostics(file):
    glassDoor = pd.read_csv(file)
    numBins = 30
    glassDoor = glassDoor.dropna()
    plt.hist(glassDoor["Ratings"],bins=numBins)
    plt.show()
def scrapeGlassData():
    door = GlassDoorScraper()
    door.scrape()
    glassDoorDiagnostics("./glassDoorData.csv")

if __name__ == "__main__":
    scrapeGlassData()
