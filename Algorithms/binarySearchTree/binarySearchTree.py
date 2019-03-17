#!/usr/bin/env python

import random

# custom error when item doesn't exist. For future usage; not used currently
class Error(Exception): pass
class ItemExistsError(Error):
    """Item in Binary Search Tree"""

class BinarySearchTreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    # prints tree as root, left child, right child
    def __str__(self):
        return str(preOrder(self))
# inserts value 'item' in binary search tree bst
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
# finds item in binary tree object bst
def traverse(bst,item):
    if bst == None:
        return False
    elif item == bst.val:
        return True
    elif item < bst.val:
        return traverse(bst.left,item)
    else:
        return traverse(bst.right,item)
# number of nodes in binary search tree
def countNodes(bst):
    if bst == None:
        return 0
    else:
        return 1 + countNodes(bst.left) + countNodes(bst.right)
# orders binary search tree in root, left child, right child
def preOrder(bst,lst=[]):
    if bst != None:
        lst += [bst.val]
        lst = preOrder(bst.left,lst)
        lst = preOrder(bst.right,lst)
    return lst
# order binary search tree left child, root, right child
def inOrder(bst,lst=[]):
    if bst != None:
        lst = inOrder(bst.left,lst)
        lst += [bst.val]
        lst = inOrder(bst.right,lst)
    return lst
# determines the binary search tree bst height (depth of deepest leaf)
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
# number of nodes without children
def countLeaves(bst):
    if bst == None:
        return 0
    else:
        # None implies no children
        if bst.left == None and bst.right == None:
            return 1
        return countLeaves(bst.left) + countLeaves(bst.right)
