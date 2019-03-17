#!/usr/bin/env python

import random

# Implementations of Linked Lists (collection using linked nodes)
# DLL Disadvantages: Requires extra space for previous reference
class Node(object):
    def __init__(self,val,reference = None):
        self.val = val
        self.reference = reference
    def __str__(self):
        return str(self.val)

# Lists are useful in that they can assemble multiple objects into one
# entity (object) - collection
def printLst(node):
    while node != None:
        print(node,end=" ")
        node = node.reference # using the reference field of the Node object,
        # we can move on to the next node in the linked list.
    print()
# Print tail backwards then head
def printReversedLst(node):
    if node == None:
        return
    head = node
    tail = node.reference
    printReversedLst(tail) # using recursion to print a list backwards
    print(head,end=" ")

# Fundamental Ambiguity Theorem - ambiguity in reference to a node.
# Reference to list node can be an object or first list node

class DoubleNode(object):
    def __init__(self,val=None,nextReference=None,prevReference=None):
        self.val = val
        self.nextReference = nextReference
        self.prevReference = prevReference
    def __str__(self):
        return "%s %s %s"%(str(self.prevReference),
                           str(self.val),
                           str(self.nextReference))

class SinglyLinkedList(object):
    def __init__(self):
        self.head = None
        self.length = 0
    # prints reversed linked list by reversing head
    def printReversed(self):
        print("[",end=" ")
        if self.head != None:
            self.head.printReversed()
    # adds element to front of linked list - new node references head
    def addFirstElem(self,value):
        node = Node(value)
        self.head = node
        node.reference = self.head
        self.length += 1

class DoublyLinkedList(object):
    def __init__(self):
        self.head = None
    # newNode references head of linked list and vice versa
    # - new head becomes the newNode
    def push(self, val):
        newNode = DoubleNode(val)
        newNode.reference = self.head
        if self.head != None:
            self.head.prevReference = newNode
        self.head = newNode
    # if the prevNode not within list, nothing executed
    def insertAfter(self,prevNode, newVal):
        if prevNode == None:
            return
        newNode = DoubleNode()
        # fits newNode references to be between prevNode & node 
        # after that 
        newNode.nextReference = prevNode.nextReference
        prevNode.nextReference = newNode
        newNode.prevReference = prevNode
        if newNode.nextReference != None:
            newNode.nextReference.prevReference = newNode
    # adds newNode to end of linked list
    def append(self, newVal):
        newNode = Node(newVal)
        newNode.nextReference = None
        # if there head is None, newNode has None reference & newNode becomes
        # the list's head
        if self.head == None:
            newNode.prevReference = None
            self.head = newNode
            return
        last = self.head
        while last.nextReference != None:
            last = last.nextReference
        last.nextReference = newNode
        newNode.prevReference = last
        return
    # prints out list of nodes
    def printList(self, node):
        # prints out list through nextReference links between el'ts
        while node.val != None:
            print("%d"%node.val,end=" ")
            last = node
            node = node.nextReference
        print()
        # linked by prior references, we print out linked list backwards
        while last != None:
            try:
                print("%d" %(last.val),end=" ")
            except:
                pass
            last = last.prevReference

class CircularLinkedList(object):
    def __init__(self):
        self.head = None
    # circular node appended to beginning of linked list
    def add(self, val):
        addedNode = CircularNode(val)
        temp = addedNode.reference = self.head
        # temporary var temp referenced until reaching head
        # - head references added node. 
        if self.head != None:
            while(temp.reference != self.head):
                temp = temp.reference
            temp.reference = addedNode
        else:
            # addedNode reference itself if no head
            addedNode.reference = addedNode
        self.head = addedNode
    # prints the value of node of "temp" until we reach back to 
    # head of list, then terminate printLst
    def printLst(self):
        temp = self.head
        if self.head != None:
            while True:
                print("{}".format(temp.val),end=" ")
                temp = temp.reference
                if temp == self.head:
                    break

def testLinkedListClass():
    node1 = Node(30)
    node2 = Node(40)
    node3 = Node(50)
    node1.reference = node2
    node2.reference = node3
    printLst(node1)
    printReversedLst(node1)

    print()
    doubleLst = DoublyLinkedList()
    doubleLst.append(10)
    for i in range(5):
        doubleLst.push(random.randint(1,50))
    doubleLst.append(23)
    doubleLst.insertAfter(doubleLst.head.nextReference, 7)
    doubleLst.printList(doubleLst.head)

    print()
    circularLst = CircularLinkedList()
    circularLst.add(23)
    circularLst.add(43)
    circularLst.add(2)
    circularLst.add(7)
    circularLst.printLst()

if __name__ == "__main__":
    testLinkedListClass()
