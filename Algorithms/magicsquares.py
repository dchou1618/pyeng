#!/usr/bin/env python

def createBoard(n):
    result = []
    for row in range(n):
        result.append([None]*n)
    return result

# n is up to a specific number n and d is common difference
# in arithmetic progression

def makeMagicSquare(n,start,d):
    arithProgress = list(range(start,n*n+1,d))
    if n%2 == 0: return None

    board = createBoard(n)
    startingPosition = n//2
    index,row,col = 0,0,startingPosition
    lastRow = True
    board[0][startingPosition] = arithProgress[index]
    index += 1

    while index < len(arithProgress):
        # follows algorithm of filling in square with subsequent number
        # by moving up-right. If not a valid move, move down to next column
        # or wraparound to first col. If moved to occupied cell, then
        # move down
        row -= 1
        col += 1
        if row < 0:
            if col >= len(board[0]):
                row += 2
                col -= 1
                board[row][col] = arithProgress[index]
            else:
                row = len(board)-1
                if board[row][col] != None:
                    row = 1
                board[row][col]=arithProgress[index]
        elif col >= len(board[0]):
            col = 0
            if board[row][col] != None:
                row += 1
            board[row][col]=arithProgress[index]
        else:
            if board[row][col] == None:
                board[row][col] = arithProgress[index]
            else:
                row += 2
                col -= 1
                board[row][col] = arithProgress[index]
        index += 1
    return board
