#!/usr/bin/env python3

def extendedEuclid(n,m):
    x1,y1,x2,y2 = 0,1,1,0
    while n != 0:
        r,m,n = m//n,n,m%n
        y1,y2 = y2, y1 - r*y2
        x1,x2 = x2, x1 - r*x2
    return m,x1,y1
