#!/usr/bin/env python

import magicsquares,Tkinter as tk, sys
from Tkinter import *

def init(data):
    data.num = 3
    data.rows = data.height/data.num
    data.cols = data.width/data.num
    data.progress = magicsquares.createBoard(data.num)
    data.colors = []
    for row in range(data.num):
        data.colors.append(["blue"]*data.num)
    data.highlighted = [data.num//2,data.num//2]
    data.magic = magicsquares.makeMagicSquare(data.num,1,1)
    data.changes = []
# arrow keys to navigate magic squares grid
def keyPressed(event,data):
    if event.keysym == "Up":
        data.highlighted[0] -= 1
        if data.highlighted[0] < 0:
            data.highlighted[0] = data.num-1
    elif event.keysym == "Down":
        data.highlighted[0] += 1
        if data.highlighted[0] >= data.num:
            data.highlighted[0] = 0
    elif event.keysym == "Left":
        data.highlighted[1] -= 1
        if data.highlighted[1] < 0:
            data.highlighted[1] = data.num - 1
    elif event.keysym == "Right":
        data.highlighted[1] += 1
        if data.highlighted[1] >= data.num:
            data.highlighted[1] = 0
    elif event.keysym == "t":
        data.num += 2
        data.changes = []
    elif event.keysym == "r":
        init(data)
        
def getCellBounds(data,row,col):
    x1 = col*data.cols
    y1 = row*data.rows
    x2 = (col+1)*data.cols
    y2 = (row+1)*data.rows
    return (x1,y1,x2,y2)

def mousePressed(event,data):
    row = int(event.y/(data.rows))
    col = int(event.x/(data.cols))
    if (row,col) in data.changes:
        index = data.changes.index((row,col))
        data.changes = data.changes[:index]+data.changes[index+1:]
    else:
        data.changes += [(row,col)]
# draws magicsquares using makeMagicSquare function in magicsquares.py        
def drawMagicBoard(canvas,data):
    data.progress = magicsquares.createBoard(data.num)
    data.colors = []
    data.rows = data.height/data.num
    data.cols = data.width/data.num
    for row in range(data.num):
        lst = []
        for col in range(data.num):
            if (row,col) in data.changes:
                lst.append("purple")
            else:
                lst.append("blue")
        data.colors.append(lst)
    data.magic = magicsquares.makeMagicSquare(data.num,1,1)
    # highlights current highlighted cell by user
    for row in range(data.num):
        for col in range(data.num):
            if row == data.highlighted[0] and col == data.highlighted[1]:
                data.colors[row][col] = "yellow"
            x1,y1,x2,y2=getCellBounds(data,row,col)
            canvas.create_rectangle(x1,y1,x2,y2,
                                    fill=data.colors[row][col],width=5)
            canvas.create_text((x1+x2)/2,(y1+y2)/2,
                                text=str(data.magic[row][col]),
                                font="Arial "+
                                str(int(min(data.rows,data.cols))//2)
                                +" bold")
def redrawAll(canvas,data):
    drawMagicBoard(canvas,data)

# generic graphics run function
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

    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 100 # milliseconds
    root = Tk()
    root.resizable(width=False, height=False) # prevents resizing window
    init(data)
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    root.mainloop()  
if __name__ == "__main__":
    try:
        width = int(sys.argv[1])
        height = int(sys.argv[2])
        run(width, height)
    except:
        pass
