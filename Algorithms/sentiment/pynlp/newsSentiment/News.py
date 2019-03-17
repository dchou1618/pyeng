#!/usr/bin/env python
# webscraping, using a couple of news API's.
import requests
# importing sentiment python file
import sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
# plotting dependencies
import seaborn as sns
import matplotlib.pyplot as plt

# initializes SentimentIntensityAnalyzer object
sentiment_analy = SentimentIntensityAnalyzer()
# opens the news at the url
def openNews(url):
    open = requests.get(url).json()
    articles = open["articles"]
    results = []
    for article in articles:
        results.append(article["description"]) # dictionary access
    return results
#     descriptions = []
#     for i in range(len(results)):
#         descriptions.append(results[i])
#     return descriptions

# using newsapi for ny times and cnn, construct list of sentiment data by taking the descriptions of the 
# news articles
def NewsScrape():
    urls = { "NYTimes": "https://newsapi.org/v2/top-headlines?sources=the-new-york-times&apiKey=298448e835324fd4bc75dd40404b1137",
             "CNN": "https://newsapi.org/v2/top-headlines?sources=cnn&apiKey=298448e835324fd4bc75dd40404b1137"}
    sentiment_data = []
    for key in urls:
       sentiment_data.append(openNews(urls[key]))
    return sentiment_data

# generates wordcloud from the text using wordcloud_draw
def wordCloudNews():
    text = ""; news = NewsScrape()
    for row in range(len(news)):
        for col in range(len(news[row])):
            try: text += news[row][col]
            except: pass
    sentiment.wordcloud_draw(text)
# displays jointplot of the positive and negative sentiment distribution in 
# news
def diagnostics(text):
    text = NewsScrape(); negatives, positives = [], []
    for row in range(len(text)):
        for col in range(len(text[row])):
                description = text[row][col]
                negatives += [sentiment_analy.polarity_scores(description)["neg"]]
                positives += [sentiment_analy.polarity_scores(description)["pos"]]
    data = pd.DataFrame()
    data["Negatives"] = positives
    data["Positives"] = negatives
    sns.jointplot(x=data["Negatives"], y=data["Positives"], kind='kde', color='blue')
    plt.legend()
    plt.xlabel("Negatives")
    plt.ylabel("Positive")
    plt.show()
if __name__ == "__main__":
    diagnostics(NewsScrape())
    wordCloudNews()
