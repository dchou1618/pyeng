#!/usr/bin/env python
import Tkinter as tk, binarySearchTree, random, math
from binarySearchTree import *
from Tkinter import *

# Takes in system arguments run in shell
rootVal = int(sys.argv[1])
numNodes = int(sys.argv[2])

def init(data):
    data.nodes = numNodes
    data.outline = 5
    data.size = 30
    data.rootLocation = [data.width//2,data.size]
    data.root = BinarySearchTreeNode(rootVal)
    # randomly generates new nodes in binary search tree until numNodes
    while countNodes(data.root) != data.nodes:
        r = random.randint(10,100)
        insertBST(data.root,r)
        
def keyPressed(event, data):
    if event.keysym == "r":
        init(data)
# draws centered node object around x,y
def drawNode(canvas,data,bst,x,y):
    xMid,yMid = (2*x+data.size)/2,(2*y+data.size)/2
    canvas.create_oval(x,y,x+data.size,y+data.size,
                       fill="gold",width=data.outline)
    canvas.create_text(xMid,yMid,text=str(bst.val))
# recursively draws binary search tree
def drawTree(canvas,data,bst,x1,x2,y):
    drawNode(canvas,data,bst,(x1+x2)/2,y)
    parentX = (x1+x2)/2
    parentY = y
    leftX = (3*x1 + x2)/4
    rightX = (x1 + 3*x2)/4
    childY = y + data.size
    
    if bst.left != None:
        # draws line to drawn left node 
        canvas.create_line(parentX + data.size,parentY + data.size,
                           leftX + data.size,childY + data.size)
        drawTree(canvas,data,bst.left,x1,(x1+x2)/2,childY)
    if bst.right != None:
        # draws line to drawn right node
        canvas.create_line(parentX + data.size,parentY + data.size,
                           rightX + data.size,childY + data.size)
        drawTree(canvas,data,bst.right,(x1+x2)/2,x2,childY)
# displays basic characteristics of tree (e.g. leaves, height)
def drawInfo(canvas,data,nodes,h,preOrdered,inOrdered,maxNode,countLeaves):
    canvas.create_text(50,data.height*0.85,anchor="w",
                       text = "Number of Nodes: "+str(nodes),
                       font="Helvetica 15 bold")
    canvas.create_text(data.width*0.30,data.height*0.85,anchor="w",
                       text = "Number of Leaves: "+str(countLeaves),
                       font="Helvetica 15 bold")
    canvas.create_text(50,data.height*0.88,anchor="w",
                       text = "Height of Tree: "+str(h),
                       font="Helvetica 15 bold")
    canvas.create_text(50,data.height*0.91,anchor="w",
                       text = "PreOrder: "+str(preOrdered),
                       font="Helvetica 15 bold")
    canvas.create_text(50,data.height*0.94,anchor="w",
                       text = "InOrder: "+str(inOrdered),
                       font="Helvetica 15 bold")
    canvas.create_text(50,data.height*0.97,anchor="w",
                       text = "Max Node Value: "+str(maxNode),
                       font="Helvetica 15 bold")
# draws the tree and basic information
def redrawAll(canvas,data):
    drawTree(canvas,data,data.root,0,data.width,data.size*2)
    h = treeHeight(data.root)
    preOrdered = preOrder(data.root,[])
    inOrdered = inOrder(data.root,[])
    maxNode = inOrdered[len(inOrdered)-1]
    numLeaves = countLeaves(data.root)
    canvas.create_rectangle(0,data.height*0.83,data.width,data.height,
                            width=data.outline)
    drawInfo(canvas,data,data.nodes,h,preOrdered,inOrdered,maxNode,numLeaves)

###############################################################################
###############################################################################
# generic graphics run function
def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    root = Tk()
    root.resizable(width=False, height=False) # prevents resizing window
    init(data)
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    redrawAllWrapper(canvas, data)
    root.mainloop()  # blocks until window is closed

if __name__ == "__main__":
    # system takes in total of 4 arguments [rootVal] [numNodes] [width] [height]
    width,height = int(sys.argv[3]),int(sys.argv[4])
    run(width,height)
