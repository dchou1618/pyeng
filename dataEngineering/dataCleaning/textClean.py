#!/usr/bin/python

# importing system and text cleaning dependencies
import sys, os, re
# printing out current files in directory
print(os.listdir("../"))

# input file is considered to be the text
text = sys.stdin

class textClean(object):
    def __init__(self,text):
        self.text = text
    def clean(self):
        pass

def textFileCleaner():
    cleanser = textClean(text)
    cleanser.clean()

if __name__ == "__main__":
    textFileCleaner()
