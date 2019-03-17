#!/usr/bin/python
# scraping tweets using twitterscraper
import sys,twitterscraper
from twitterscraper import query_tweets
def scrapeTweets(key=sys.argv[1]):
    # query with limit of 20 
    tweetsLst = query_tweets(key,20)
    file = open("output.txt","w")
    for tweet in query_tweets(key,20):
        # message encoded in utf-8 to handle other languages
        message = tweet.text.encode("utf-8")
        # parse message into phrases in list
        phrases = message.split()
        # basic removal of non-clean phrases
        for phrase in range(len(phrases)-1,-1,-1):
            if "http" in phrases[phrase]:
                phrases.pop(phrase)
        file.write(" ".join(phrases))
if __name__ == "__main__":
    scrapeTweets()
