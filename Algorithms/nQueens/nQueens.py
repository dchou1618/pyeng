#!/usr/bin/env python

# creates n by n board
def createBoard(n):
    result = []
    for row in range(n):
        result.append([False]*n)
    return result
# obtains the diagonals from the board given a row and column
def getDiagonal(board,row,col):
    # finding el'ts along row above entry
    origRow,origCol=row,col
    diagonal = [board[row][col]]
    # iterates over diagonal by row, col until beyond bounds
    while True:
        try:
            row += 1
            col += 1
            if row >= len(board) or col >= len(board[0]):
                raise Exception
            diagonal += [board[row][col]]
        except:
            break
    # goes through other diagonal until beyond bounds
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
    # given the row and column are within bounds, iterate through row, col to construct
    # diagonal on board
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
            # determines if any multiple queens in same row, col or diagonal
            lineFire = d.count(True)>1 or r.count(True)>1 or c.count(True)>1
            if a[row][col] and lineFire:
                return False
    return True
