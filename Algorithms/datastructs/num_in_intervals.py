from sortedcontainers import SortedList
import random
import time

# we use the abstracted library heapq
import heapq as hq

def inefficient_coverage_func(intervals, nums):
	start = time.time()
	for i,num in enumerate(nums):
		covered = 0
		for lo, hi in intervals:
			if num >= lo and num <= hi:
				covered += 1
		nums[i] = covered
	end = time.time()
	print(f"{end-start} seconds to run inefficient.")
	return nums

def identify_coverage(intervals, nums):
	"""
	identify_coverage - given a list of intervals, return a list with length equal
	to the length of nums, where each entry denotes how many intervals that number 
	lies within. We use the SortedList struct for this problem.
	|I| = length of intervals
	|N| = length of nums
	* sweep line algorithm - one pass through, but keep track 
	of how many lamps have started and how many have ended.
	"""
	start_time = time.time()
	lamps_active = 0
	# process the points in priority order of 
	# 1 - note the active lamps
	# 2 - encounter point
	# 3 - stop lamp activity
	heap_lst = []
	# O(|I|)
	# we have a min heap by default
	for start, end in intervals:
		# start denoted by -1
		heap_lst.append((start, -1))
		# end is the last priority
		heap_lst.append((end, 1))
	# O(|N|)
	for i, num in enumerate(nums):
		heap_lst.append((num, 0, i))
	hq.heapify(heap_lst)
	print(heap_lst)
	# O(|I|+|N|)
	while len(heap_lst) > 0:
		val = hq.heappop(heap_lst)
		if val[1] == -1:
			lamps_active+=1
		elif val[1] == 1:
			lamps_active-=1
		else:
			# we have a num
			nums[val[2]] = lamps_active
	end_time = time.time()
	print(f"Took {end_time-start_time} seconds to run efficient.")
	return nums

if __name__ == "__main__":
	intervals = [[random.randint(1, 10000), random.randint(10001, 20000)] for _ in range(10000)]
	numbers = [random.randint(1, 20000) for _ in range(2000)]
	#intervals = [[1,7],[5,11],[7,9]]
	#numbers = [1,5,7,9,10,15]
	coverages1 = identify_coverage(intervals, numbers)
	coverages2 = inefficient_coverage_func(intervals, numbers)
	#print(coverages2)
	print(coverages1 == coverages2)


