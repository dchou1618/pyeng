#!/usr/bin/env python3

'''
sievePrime Generation
'''
import random, numpy as np, math

def sieve(n):
    primality = [True]*(n+1)
    primality[0] = primality[1] = False
    nPrimes = []
    for prime in range(n+1):
        if primality[prime]:
            nPrimes.append(prime)
            for multiple in range(2*prime,n+1,prime):
                primality[multiple] = False
    return nPrimes

def isPrime(p):
    if p < 2:
        return False
    elif p == 2:
        return True
    elif p%2 == 0:
        return False
    for factor in range(3,math.ceil(math.sqrt(p)),2):
        if p%factor == 0:
            return False
    return True
