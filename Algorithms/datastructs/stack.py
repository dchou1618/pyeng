#!/usr/bin/env python
import sys, random
from pythonds.basic.stack import Stack
# Stack Data Structure
class Stack(object):
    def __init__(self):
        self.elems = []
    def isEmpty(self):
        return len(self.elems) == 0
    def pop(self):
        return self.elems.pop()
    def push(self,newValue):
        self.elems.append(newValue)
    def peek(self):
        """
        finds the last element of the stack object
        """
        end = len(self.elems)-1
        return self.elems[end]
    def find(self,val):
        """
        binary search to determine if value in stack
        """
        sortedLst = sorted(self.elems)
        print("Sorted Stack to Look Into:", sortedLst)
        lo,hi = 0,len(self.elems)
        result = []
        while lo <= hi:
            mid = (lo+hi)//2
            if sortedLst[mid] == val:
                result.append(sortedLst[mid])
                return "Found: " + str(result)
            elif sortedLst[mid] < val:
                result.append(sortedLst[mid])
                lo = mid + 1
            else:
                result.append(sortedLst[mid])
                hi = mid - 1
        return "Not Found: " + str(None)
    def quickFind(self,val):
        return val in set(self.elems)
    def size(self):
        return len(self.elems)
    def __repr__(self):
        return str(self.elems)

def testStack():
    stack1 = Stack()
    assert(stack1.isEmpty())
    for i in range(10):
        stack1.push(random.randint(1,100))
    print("First Stack:",stack1)

if __name__ == "__main__":
    testStack()
