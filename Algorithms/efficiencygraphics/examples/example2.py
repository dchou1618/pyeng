#!/usr/bin/python

import sys, random, time

def randomLstVal(n):
    result = []
    for i in range(n):
        r = random.randint(1,n)
        result += [r]
    return result,random.randint(1,n)

def getCounts(L):
    counts = dict()
    for val in L:
        counts[val] = counts.get(val,0) + 1
    return counts

def getPairSum(lst, target):
    start = time.time()
    pairs = dict()
    for elem in lst:
        pairs[elem] = target-elem # stores el'ts & difference in dictionary
    counts = getCounts(lst)
    for key in pairs:
        if pairs[key] in lst:
            if pairs[key] == key and counts[key] < 2: return time.time()-start
            else: return time.time()-start
    return time.time()-start

def pairsTimesSizes():
    runTimeInputSizes = open("pairsTimeInput.txt","w+")
    data = """"""
    try:
        beginningSize = int(sys.argv[1])
        endingSize = int(sys.argv[2])
        stepSize = int(sys.argv[3])
        for i in [val for val in range(beginningSize,endingSize,stepSize)]:
            L,target = randomLstVal(i)
            runTime = getPairSum(L,target)
            data += str(i) + " " + str(runTime)
            data += "\n"
    except:
        print("Not enough inputs or input order")
    runTimeInputSizes.write(data)
    return data

if __name__ == "__main__":
    pairsTimesSizes()
