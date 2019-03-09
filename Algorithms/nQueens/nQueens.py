#!/usr/bin/env python

def createBoard(n):
    result = []
    for row in range(n):
        result.append([False]*n)
    return result

def getDiagonal(board,row,col):
    # finding el'ts along row above entry
    origRow,origCol=row,col
    diagonal = [board[row][col]]
    while True:
        try:
            row += 1
            col += 1
            if row >= len(board) or col >= len(board[0]):
                raise Exception
            diagonal += [board[row][col]]
        except:
            break
    row,col=origRow,origCol
    while True:
        try:
            row += 1
            col -= 1
            if row >= len(board) or col < 0:
                raise Exception
            diagonal += [board[row][col]]
        except:
            break
    if 0 < row < len(board) and 0 < col < len(board[0]):
        row,col = origRow,origCol
        while True:
            try:
                row += 1
                col += 1
                if row >= len(board) or col >= len(board[0]):
                    raise Exception
                diagonal += [board[row][col]]
            except:
                break
    return diagonal

def getRow(board,row):
    return board[row]

def getCol(board,col):
    result = []
    for row in board:
        if col < len(row):
            result.append(row[col])
    return result

def nQueensChecker(a):
    for row in range(len(a)):
        for col in range(len(a[0])):
            d = getDiagonal(a,row,col)
            r = getRow(a, row)
            c = getCol(a, col)
            lineFire = d.count(True)>1 or r.count(True)>1 or c.count(True)>1
            if a[row][col] and lineFire:
                return False
    return True
