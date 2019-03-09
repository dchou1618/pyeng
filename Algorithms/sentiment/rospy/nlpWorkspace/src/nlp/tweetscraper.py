#!/usr/bin/python
# scraping tweets using twitterscraper
import sys,twitterscraper
from twitterscraper import query_tweets
def scrapeTweets(key=sys.argv[1]):
    tweetsLst = query_tweets(key,20)
    file = open("output.txt","w")
    for tweet in query_tweets(key,20):
        #file.write(tweet.text.encode("utf-8"))
        message = tweet.text.encode("utf-8")
        phrases = message.split()
        for phrase in range(len(phrases)-1,-1,-1):
            if "http" in phrases[phrase]:
                phrases.pop(phrase)
        file.write(" ".join(phrases))
if __name__ == "__main__":
    scrapeTweets()