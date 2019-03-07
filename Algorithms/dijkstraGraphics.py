#!/usr/bin/env python

import dijkstra,math,sys,random,Tkinter as tk
from Tkinter import *
from dijkstra import *

def createNetwork(outerLoop=int(sys.argv[1]),innerLoop=int(sys.argv[2])):
    letters = string.ascii_uppercase
    L = []
    items = []
    for source in range(outerLoop):
        for dest in range(source+1,innerLoop):
            r = random.randint(1,20)
            items += [letters[source],letters[dest]]
            L += [Edge(letters[source],letters[dest],r)]
            r = random.randint(1,20)
            items += [letters[dest],letters[source]]
            L += [Edge(letters[dest],letters[source],r)]
    network = Network(L)
    start = random.randint(0,len(L)-1)
    end = random.randint(0,len(L)-1)
    dijkstraLst = network.dijkstra(L[start].sourceNode,L[end].sourceNode)
    return items,L,dijkstraLst

def init(data):
    data.nodes,data.connections,data.dijkstra = createNetwork(12,11)
    data.cx,data.cy = data.width/2,data.height/2
    data.radius = min(data.width*0.40,data.height*0.40)

    data.size = 20
    data.showPath = False
    data.outline = 3
    data.nodeTuples = dict()
    data.nodes = list(set(data.nodes))

    data.color = "gold"
    data.colors = {data.nodes[i]:data.color for i in range(len(data.nodes))}
def timerFired(data): pass
def mousePressed(event,data): pass
def keyPressed(event,data):
    if event.keysym == "r":
        init(data)
    elif event.keysym == "space":
        data.showPath = not data.showPath
        data.colors = {data.nodes[i]:data.color for i in range(len(data.nodes))}
def drawNode(canvas,data,cx,cy,r,label,i):
    canvas.create_oval(cx-r,cy-r,cx+r,cy+r,
                       fill=data.colors[data.nodes[i]],width=data.outline)
    canvas.create_text(cx,cy,text=str(label))
def drawBasicNetwork(canvas,data):
    initialAngle = 0
    dA = 2*math.pi/len(data.nodes)
    for i in range(len(data.nodes)):
        x = data.cx+data.radius*math.cos(initialAngle)
        y = data.cy-data.radius*math.sin(initialAngle)
        drawNode(canvas,data,x,y,data.size,data.nodes[i],i)
        data.nodeTuples[data.nodes[i]] = [x,y]
        initialAngle += dA
def drawConnections(canvas,data):
    for edge in data.connections:
        sourceCoordinates = data.nodeTuples[edge.sourceNode]
        destinationCoordinates = data.nodeTuples[edge.destinationNode]
        x1,y1=sourceCoordinates
        x2,y2=destinationCoordinates
        canvas.create_line(x1,y1,x2,y2,width=data.outline)
def drawPath(canvas,data):
    for i in range(1,len(data.dijkstra)):
        letter = data.dijkstra[i]
        prevLetter = data.dijkstra[i-1]
        x1,y1=data.nodeTuples[letter]
        x2,y2=data.nodeTuples[prevLetter]
        canvas.create_line(x1,y1,x2,y2,fill="blue",width=data.outline)
    try:
        data.colors[data.dijkstra[0]]= "red"
        data.colors[data.dijkstra[i]] = "green"
    except:
        init(data)
def redrawAll(canvas,data):
    drawBasicNetwork(canvas,data)
    drawConnections(canvas,data)
    if data.showPath: drawPath(canvas,data)
    drawBasicNetwork(canvas,data)
##############################################################################
##############################################################################
def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    root = Tk()
    root.resizable(width=False, height=False) # prevents resizing window
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    redrawAllWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
if __name__ == "__main__":
    width,height = int(sys.argv[3]),int(sys.argv[4])
    run(width, height)
