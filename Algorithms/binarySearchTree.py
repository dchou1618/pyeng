#!/usr/bin/env python

import random

class Error(Exception): pass
class ItemExistsError(Error):
    """Item in Binary Search Tree"""

class BinarySearchTreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    def __str__(self):
        return str(preOrder(self))
        
def insertBST(bst,item):
    if bst == None:
        return BinarySearchTreeNode(item)
    elif item > bst.val:
        bst.right = insertBST(bst.right,item)
    elif item < bst.val:
        bst.left = insertBST(bst.left,item)
    else:
        return bst
    return bst

def traverse(bst,item):
    if bst == None:
        return False
    elif item == bst.val:
        return True
    elif item < bst.val:
        return traverse(bst.left,item)
    else:
        return traverse(bst.right,item)

def countNodes(bst):
    if bst == None:
        return 0
    else:
        return 1 + countNodes(bst.left) + countNodes(bst.right)

def preOrder(bst,lst=[]):
    if bst != None:
        lst += [bst.val]
        lst = preOrder(bst.left,lst)
        lst = preOrder(bst.right,lst)
    return lst

def inOrder(bst,lst=[]):
    if bst != None:
        lst = inOrder(bst.left,lst)
        lst += [bst.val]
        lst = inOrder(bst.right,lst)
    return lst

def treeHeight(bst):
    if bst == None:
        return 0
    else :
        leftHeight = treeHeight(bst.left)
        rightHeight = treeHeight(bst.right)

        if leftHeight > rightHeight:
            return leftHeight+1
        else:
            return rightHeight+1

def countLeaves(bst):
    if bst == None:
        return 0
    else:
        if bst.left == None and bst.right == None:
            return 1
        return countLeaves(bst.left) + countLeaves(bst.right)
