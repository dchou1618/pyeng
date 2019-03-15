#!/usr/bin/env python3

'''
Euclid's Algorithm - GCD
'''

def gcd(n,m):
    n,m = abs(n),abs(m)
    while n != 0 and m != 0:
        if n>m: n,m = m,n%m
        elif n<m: n,m = m%n,n
        else: return n
    return m or n # one will be 0, which won't be returned

def lcm(n,m):
    return (n*m)/gcd(n,m)
