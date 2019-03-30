#/usr/bin/env python3

# countriesScrape.py

# dependencies
import requests, re
from bs4 import BeautifulSoup

url = "https://www.dfa.ie/travel/travel-advice/a-z-list-of-countries/"
headers = {'user-agent': 'Chrome/60.0.3112.90'}

def readFile(path):
    with open(path,"r") as f:
        content = f.read().splitlines()
    return content

class CountriesScrape:
    def scrapeForCountries(self):
        countryLoc = dict()
        countries = readFile("countries.txt")
        for country in countries:
            country = country.split()
            countryLoc[" ".join(country[3:])] = (country[1],country[2])
        return countryLoc
        # extracted from list of countries on earth
        # response = requests.get(url,headers=headers)
        # listCountriesHTML = BeautifulSoup(response.text,"html.parser")
        # countriesList = listCountriesHTML.find_all("div",class_ = "mcol")
        # unformattedCountries = countriesList[0].ul.text
        # countriesLst = re.findall("[A-Z][^A-Z]*",unformattedCountries)
        # return countriesLst

if __name__ == "__main__":
    c = CountriesScrape()
    print(c.scrapeForCountries())
