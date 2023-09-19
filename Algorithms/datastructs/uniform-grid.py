# Supposedly, recursion can be employed here.

from sortedcontainers import SortedDict
import copy
def in_bounds(i,j, field):
	return i >= 0 and i < len(field) and j >= 0 and j < len(field[0])


def get_moves(i,j,direction):
	if in_bounds(i+dx,j+dy, field) and field[i+dx][j+dy] != "#":
		return [(i+dx,j+dy)]
	return []


def drop_in_place(i,j, field):
	# if the position in the field is already # a block, then we 
	# don't have any where to go and that's the most we can place.
	if field[i][j] == "#":
		return field, False
	while in_bounds(i,j+1, field) and field[i][j+1] != "#":
		j += 1
	# then start going downward
	while in_bounds(i+1,j, field) and field[i+1][j] != "#":
		i += 1
	field[i][j] = "#"
	return field, True

def print_matrix(field):
	for row in field:
		print(row)

def num_moves(curr_row, field, find_mx_moves):
	i = 0
	total_moves = 0
	while curr_row >= 0 and curr_row < len(field):
		field, success = drop_in_place(curr_row, 0, field)
		if not success:
			if find_mx_moves:
				curr_row -= 1
			else:
				# to find the minimum, you begin from the top and move down.
				curr_row += 1
		else:
			total_moves += 1
	return total_moves


def minmax_grid(field):
	field1 = copy.deepcopy(field)
	min_moves = num_moves(0, field1, False)
	field2 = copy.deepcopy(field)
	max_moves = num_moves(len(field)-1, field2, True)
	return min_moves, max_moves



if __name__ == "__main__":
	print(minmax_grid([[".","#","#"],
		["#",".","."],
		[".", ".", "."]]))

	# should be [4,4]

	print(minmax_grid([[".","#","#"],
		[".",".","#"],
		[".", ".", "."]]))
	# should be [3,6]

	print(minmax_grid([[".",".",".", "."],
		[".",".",".", "."],
		[".", ".", ".", "."]]))
	# should be [12,12]



