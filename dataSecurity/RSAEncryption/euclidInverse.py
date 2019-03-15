#!/usr/bin/env python3

from extendedEuclid import *
'''
Modular Multiplicative Inverse
'''
def multInverse(n,m):
    g,x,_ = extendedEuclid(n,m)
    if g == 1:
        return x%m
