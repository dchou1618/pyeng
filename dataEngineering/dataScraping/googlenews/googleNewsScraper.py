#!/usr/bin/env python3

# import dependencies
import matplotlib.pyplot as plt,time,random,requests,pandas as pd,numpy as np
import nltk, seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')
from wordcloud import *
from PIL import Image
from bs4 import BeautifulSoup
# visualizations
# import geopandas as gpd
# import pysal as ps
# from pysal import contrib
# import mapping as maps

headers = {'user-agent': 'Chrome/60.0.3112.90'}
# basic scraper to analyze the sentiment of articles on google news homepage

class GoogleNewsScraper:
    def __init__(self,url):
        self.url = url
    # single page inspection - text mining
    def findHeadlines(self):
        headLines,numRequests = [],0
        url = self.url
        start = time.time()
        pageResponse = requests.get(url,headers=headers)
        # numRequests += 1
        if pageResponse.status_code != 200:
            warn("Bad status code: %d"%(pageResponse.status_code))
        googleHTML = BeautifulSoup(pageResponse.text,"html.parser")
        headLinesClass = "xrnccd F6Welf R7GTQ keNKEd j7vNaf"
        mainHeadlines = googleHTML.find_all("div",class_ = headLinesClass)
        subNewsClass = "SbNwzf"
        subNewsTitles = googleHTML.find_all("div",class_ = subNewsClass)
        for articleIndex in range(len(mainHeadlines)+len(subNewsTitles)):
            if articleIndex < len(mainHeadlines):
                headlineTxt=mainHeadlines[articleIndex].article.h4.a.text
                headLines.append(headlineTxt)
            else:
                adjustedIndex = articleIndex-len(mainHeadlines)
                subNewsTxt=subNewsTitles[adjustedIndex].article.h4.a.text
                headLines.append(subNewsTxt)
        googleNewsHeadlines = pd.DataFrame({"headlines":headLines})
        googleNewsHeadlines.to_csv("googleNewsHeadlines.csv")
        self.headlines = headLines
        return headLines

    def getTopPolarityWords(self):
        polarities = {}
        for headline in self.headlines:
            words = headline.split()
            for word in words:
                wordSenti = self.sentiment(word)
                polarities[wordSenti] = polarities.get(wordSenti,[]) + [word]
        self.polarities = polarities
        return polarities

    def sentiment(self,headline):
        senti = SentimentIntensityAnalyzer()
        positivity = senti.polarity_scores(headline)
        polarities = {positivity["neg"]:"Negative",
                      positivity["neu"]:"Neutral",
                      positivity["pos"]:"Positive"}
        mostProbable = max(positivity["neg"],
                           max(positivity["neu"],positivity["pos"]))
        conclusion = polarities[mostProbable]
        return conclusion

    def drawPlots(self,text,mask,color = 'white'):
        words = " ".join([word for word in text.split()])
        self.getTopPolarityWords()
        positives = self.polarities["Positive"]
        negatives = self.polarities["Negative"]
        wordcloud1 = WordCloud(max_words = 100,
                              stopwords=STOPWORDS,
                              background_color=color,
                              width=100,height=100,
                              collocations = False,
                              mask=mask).generate(words)
        wordcloud2 = WordCloud(max_words = 100,
                              stopwords=STOPWORDS,
                              background_color=color,
                              width=100,height=100,
                              collocations = False,
                              mask=mask).generate(" ".join(positives))
        wordcloud3 = WordCloud(max_words = 100,
                              stopwords=STOPWORDS,
                              background_color=color,
                              width=100,height=100,
                              collocations = False,
                              mask=mask).generate(" ".join(negatives))
        # matches text to colors from original mask
        cloudColors = ImageColorGenerator(mask)
        # constructing subplots
        fig = plt.figure(figsize=(8,8))
        rows,columns = 2,2
        fig.add_subplot(rows,columns,1)
        # displaying plot of basic sentiment distribution
        lstSenti = [self.sentiment(headline) for headline in self.headlines]
        sentiments = pd.DataFrame({"Sentiment":lstSenti})
        sns.countplot(x="Sentiment",data=sentiments)
        plt.title("Google News Sentiment")

        # displaying word cloud of overall headlines
        fig.add_subplot(rows,columns,2)
        plt.imshow(wordcloud1.recolor(color_func=cloudColors),
                   interpolation='bilinear')
        plt.title("Google News Headlines")

        # display most common positive words
        fig.add_subplot(rows,columns,3)
        plt.imshow(wordcloud2.recolor(color_func=cloudColors),
                   interpolation='bilinear')
        plt.xlabel("Google News Positive")

        # display most common negative words
        fig.add_subplot(rows,columns,4)
        plt.imshow(wordcloud3.recolor(color_func=cloudColors),
                   interpolation='bilinear')
        plt.xlabel("Google News Negative")
        plt.show()
    def pageDiagnostics(self):
        text = " ".join(self.headlines)
        mask = np.array(Image.open("GoogleG.png"))
        self.drawPlots(text,mask)

def scrapeGoogleNews():
    mainpage = "https://news.google.com/?tab=wn&hl=en-US&gl=US&ceid=US:en"
    gNews = GoogleNewsScraper(mainpage)
    gNews.findHeadlines()
    gNews.pageDiagnostics()

if __name__ == "__main__":
    scrapeGoogleNews()
