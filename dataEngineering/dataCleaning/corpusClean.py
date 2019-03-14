#!/usr/bin/python

import pandas as pd, csv, sys
from textclean import *

corpusData = pd.read_csv(sys.stdin)

class corpusClean(textClean):
    def __init__(self,corpus = corpusData):
        super().__init__(self,corpus)
    def clean(self):
        pass

def corpusCleaner():
    cCleanser = corpusClean()
    cCleanser.clean()
if __name__ == "__main__":
    corpusCleaner()
