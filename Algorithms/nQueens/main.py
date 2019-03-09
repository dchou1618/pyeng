#!/usr/bin/env python

import Tkinter as tk, nQueens, sys
from nQueens import *
from Tkinter import *

def init(d):
    d.num = 3
    d.rows = d.cols = d.num
    d.cellWidth = d.width/d.cols
    d.cellHeight = d.height/d.rows
    d.board = createBoard(d.num)
def getCellBounds(d,r,c):
    cellWidth = d.cellWidth
    x1 = c*cellWidth
    y1 = r*cellWidth
    x2 = (c+1)*cellWidth
    y2 = (r+1)*cellWidth
    return x1,y1,x2,y2
def keyPressed(e,d):
    if e.keysym == "t":
        d.num += 1
        d.rows = d.num
        d.cols = d.num
        d.cellWidth = d.width/d.cols
        d.cellHeight = d.height/d.rows
        d.board = createBoard(d.num)
    elif e.keysym == "r":
        init(d)
def mousePressed(e,d):
    for r in range(d.rows):
        for c in range(d.cols):
            x1,y1,x2,y2 = getCellBounds(d,r,c)
            if x1 <= e.x <= x2 and y1 <= e.y <= y2:
                d.board[r][c] = not d.board[r][c]

def drawPiece(c,d,row,col):
    x1,y1,x2,y2 = getCellBounds(d,row,col)
    divisor = 10
    margin = min(abs(x1-x2),abs(y1-y2))//divisor
    queenWidth = 5
    c.create_oval(x1+margin,y1+margin,x2-margin,y2-margin,
                            fill="gold",outline="yellow",width=queenWidth)
    c.create_text((x1+x2)/2,(y1+y2)/2,text="Q",
                   font="Helvetica "+str(int(margin*2))+" bold")

def drawBoard(c,d):
    width = 5
    for row in range(d.rows):
        for col in range(d.cols):
            color = "white"
            if (row+col)%2!= 0: color = "black"
            if not nQueensChecker(d.board):
                color = "red"
            x1,y1,x2,y2 = getCellBounds(d,row,col)
            c.create_rectangle(x1,y1,x2,y2,fill=color,width = width)

def redrawAll(c,d):
    drawBoard(c,d)
    for row in range(d.rows):
        for col in range(d.cols):
            if d.board[row][col]:
                drawPiece(c,d,row,col)

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
    # and launch the app
    root.mainloop()  # blocks until window is closed

if __name__ == "__main__":
    run(int(sys.argv[1]),int(sys.argv[2]))
