#!/usr/bin/python

import sys, time, random

# assume L is a list of length N
def bigO1(L):
    start = time.time()
    N = len(L)
    i = 3
    while(i**3 < N):
        i += 3
    return time.time()-start

def randomLst(n):
    result = []
    for i in range(n):
        r = random.randint(1,n)
        result += [r]
    return result

def bigO1TimesSizes():
    runTimeInputSizes = open("bigO1TimeInput.txt","w+")
    data = """"""
    try:
        initial = int(sys.argv[1])
        end = int(sys.argv[2])
        step = int(sys.argv[3])
        for i in [val for val in range(initial,end,step)]:
            L = randomLst(i)
            runTime = bigO1(L)
            data += str(i) + " " + str(runTime)
            data += "\n"
    except:
        print("Not enough inputs or input order")
    runTimeInputSizes.write(data)
    return data

if __name__ == "__main__":
    bigO1TimesSizes()
