from collections import defaultdict
from math import sqrt


class StackNode:
	value = None
	next = None
	
	def __init__(self, value):
		self.value = value


class Stack:
	head: StackNode = None
	
	def __init__(self):
		pass
	
	def push(self, element):
		if self.head is None:
			self.head = StackNode(element)
		else:
			tmp = self.head
			self.head = StackNode(element)
			self.head.next = tmp
	
	def pop(self):
		if self.head is not None:
			tmp = self.head
			self.head = self.head.next
			return tmp.value
		return None
	
	def peek(self):
		return self.head.value


def check_route(node_a, node_b):
	pass


def test_queue():
	s = Stack()
	s.push(13)
	s.push(10)
	s.push(10)
	print(s.pop())
	print(s.pop())
	print(s.pop())
	print(s.pop())


def binary_sort(sequence):
	i = 0
	j = len(sequence) - 1
	while i < j and i < len(sequence) and j > 0:
		if sequence[i] == 0 and sequence[j] == 1:
			i += 1
			j -= 1
		elif sequence[i] == 1 and sequence[j] == 0:
			sequence[i] = 0
			sequence[j] = 1
			i += 1
			j -= 1
		elif sequence[i] == 1 and sequence[j] == 1:
			j -= 1
		elif sequence[i] == 0 and sequence[j] == 0:
			i += 1
	return sequence


def count_islands(matrix):
	count = 0
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if matrix[i][j] == 1:
				count += 1
				break_island(matrix, i, j)
	return count


def break_island(matrix, i, j):
	if 0 <= i < len(matrix) and 0 <= j < len(matrix[0]):
		if matrix[i][j] == 1:
			matrix[i][j] = 0
			break_island(matrix, i + 1, j)
			break_island(matrix, i - 1, j)
			break_island(matrix, i, j + 1)
			break_island(matrix, i, j - 1)
	else:
		return None


def test_count_island():
	mat = [[0, 1, 0, 1],
		   [1, 0, 0, 1],
		   [1, 0, 1, 0],
		   [0, 1, 1, 0]]
	print(count_islands(mat))


def find_maximal_subsequence(sequence):
	i = len(sequence) - 1
	cur_size = 0
	cur_list = []
	sums = []
	list_of_sahar = []
	if sequence is None:
		return None
	if len(sequence) == 1:
		return [sequence[0]]
	while i > 0:
		if sequence[i] > sequence[i - 1]:
			cur_size += 1
			cur_list.append(sequence[i])
		else:
			cur_size = 0
			cur_list.append(sequence[i])
			list_of_sahar.append(cur_list[:])
			sums.append(sum(cur_list))
			cur_list = []
		i -= 1
		if i == 0:
			cur_list.append(sequence[i])
			list_of_sahar.append(cur_list[:])
			sums.append(sum(cur_list))
	big = max(sums)
	
	print(list_of_sahar)
	return big


def climbStairs(n):
	"""
    :type n: int
    :rtype: int
    """
	count = 0
	table = dict()
	count += climbStairsHelper(n, table)
	return count


def climbStairsHelper(n, table):
	curCount = 0
	if n == 0:
		curCount += 1
	elif n < 0:
		return 0
	else:
		if n - 1 in table.keys():
			curCount += table[n - 1]
		else:
			res = climbStairsHelper(n - 1, table)
			curCount += res
			table[n - 1] = res
		if n - 2 in table.keys():
			curCount += table[n - 2]
		else:
			res = climbStairsHelper(n - 2, table)
			curCount += res
			table[n - 2] = res
	return curCount


def test_find_max_subseq():
	sequence = [60000, 60001, 0, 2, 65, 9, 50000, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24, 25, 26, 27,
				29, 30, 31, 32, 33, 34, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1]
	print(find_maximal_subsequence(sequence))


def maxProfit(prices):
	"""
    :type prices: List[int]
    :rtype: int
    """
	maximum = 0
	curSum = 0
	curAdditions = []
	differences = reduceDifference(prices)
	if prices is None or prices == []:
		return 0
	for i in differences:
		curSum += i
		curAdditions.append(curSum)
		if curSum < 0:
			curSum = 0
			if len(curAdditions) != 0:
				maximum = max(max(curAdditions), maximum)
			curAdditions = []
	if len(curAdditions) != 0:
		maximum = max(max(curAdditions), maximum)
	return maximum


def reduceDifference(prices):
	newList = []
	for i in range(0, len(prices) - 1):
		newList.append(prices[i + 1] - prices[i])
	return newList


def test_max_price():
	prices1 = [7, 1, 5, 3, 6, 4]
	prices2 = [7, 6, 4, 3, 1]
	print(maxProfit(prices1))
	print(maxProfit(prices2))


def rob(houses):
	current_sum = 0
	table = []
	dct = dict()
	i = 0
	houses = [k for k in houses if k != 0]
	rob_helper(houses, current_sum, table, dct, i)
	if len(houses) == 0:
		return 0
	# print(len(dct))
	return max(table)


def rob_helper(houses, current_sum, table, dct, i):
	if len(houses) < 0:
		return
	if len(houses) == 0:
		table.append(current_sum)
		return current_sum
	else:
		if (i, 2, current_sum) in dct.keys():
			pass
		else:
			dct[(i, 2, current_sum)] = rob_helper(houses[2:], current_sum + houses[0], table, dct, i + 2)
		if (i, 1, current_sum) in dct.keys():
			pass
		else:
			dct[(i, 1, current_sum)] = rob_helper(houses[1:], current_sum, table, dct, i + 1)
		return max(dct[(i, 1, current_sum)], dct[(i, 2, current_sum)])


def test_rob():
	smaller_list = [183, 219, 57, 193, 94, 233, 202, 154, 65, 240, 97, 234, 100, 249, 186, 66, 90, 238, 168, 128, 177,
					235, 50, 81, 185, 165, 217, 207, 88, 80, 112, 78, 135, 62, 228, 247, 211]
	print(rob(smaller_list))
	print(robber(smaller_list))


def robber(houses):
	dp = houses[:]
	if len(houses) > 2:
		dp[2] = dp[0] + houses[2]
		for i in range(3, len(houses)):
			dp[i] = max(dp[i - 2], dp[i - 3]) + houses[i]
	return max(dp) if len(dp) > 0 else 0


class Solution(object):
	def maxSubArray(self, arr):
		dp = [0] * len(arr)
		dp[0] = 0 if arr[0] < 0 else arr[0]
		curSum = 0 if arr[0] < 0 else arr[0]
		if all(k < 0 for k in arr):
			return max(arr)
		for i in range(1, len(dp)):
			curSum += arr[i]
			if curSum <= 0:
				dp[i] = arr[i]
				curSum = 0 if arr[i] < 0 else arr[i]
			else:
				dp[i] = curSum
		return max(dp)


def test_kadanes():
	arr = [1000, -2, 1, -3, 4, -1, 2, 1, -5, 4, 300, -304, 500]
	sol = Solution()
	print(sol.maxSubArray(arr))


def merge_On_space(self, nums1, m, nums2, n):
	"""
    :type nums1: List[int]
    :type m: int
    :type nums2: List[int]
    :type n: int
    :rtype: None Do not return anything, modify nums1 in-place instead.
    """
	i = 0
	j = 0
	newArr = []
	while i + j < m + n:
		if i < m and j < n:
			if nums1[i] <= nums2[j]:
				newArr.append(nums1[i])
				i += 1
			else:
				newArr.append(nums2[j])
				j += 1
		if j == n:
			if i < m:
				newArr.append(nums1[i])
			i += 1
		if i == m:
			if i < m:
				newArr.append(nums2[j])
			j += 1
	return newArr


def merge(self, nums1, m, nums2, n):
	"""
    :type nums1: List[int]
    :type m: int
    :type nums2: List[int]
    :type n: int
    :rtype: None Do not return anything, modify nums1 in-place instead.
    """
	while n > 0:
		if m <= 0 or nums1[m - 1] >= nums2[n - 1]:
			nums1[m + n - 1] = nums1[m - 1]
			m -= 1
		else:
			nums1[m + n - 1] = nums2[n - 1]
			n -= 1
	return nums1


def test_merge():
	# first = [1, 3, 5, 7, 9, 10, 12, 13, 15, 16, 17, 19]
	# arr2 = [0, 2, 4, 6, 8, 10, 12, 13, 15, 16]
	first = [1, 2, 3]
	arr2 = [2, 5, 6]
	m = len(first)
	n = len(arr2)
	arr1 = [0] * (m + n)
	arr1[:m] = first
	# print(merge_On_space(None, arr1, m, arr2, n))
	print(merge(None, arr1, m, arr2, n))


def isBadVersion(middle):
	if middle < 100:
		return False
	return True


def binary_search(start, end):
	middle = (start + end) // 2
	if start - end == 0:
		return start
	if isBadVersion(middle):
		return binary_search(start, middle)
	else:
		return binary_search(middle + 1, end)


def firstBadVersion(self, n):
	"""
    :type n: int
    :rtype: int
    """
	start = 0
	return binary_search(start, n)


def removeDuplicates(self, nums):
	"""
    :type nums: List[int]
    :rtype: int
    """
	st = set()
	length = len(nums)
	i = 0
	while i < length:
		while i + 1 < length and nums[i] == nums[i + 1]:
			nums.pop(i + 1)
			length -= 1
		i += 1
	return len(nums)


def rotate(self, nums, k):
	"""
    :type nums: List[int]
    :type k: int
    :rtype: None Do not return anything, modify nums in-place instead.
    """


def singleNumber(self, nums):
	"""
    :type nums: List[int]
    :rtype: int
    """
	current = nums[0]
	for i in range(1, len(nums)):
		current ^= nums[i]
	return current


def partition(nums, start, end):
	i = start - 1
	j = start
	pivot = nums[end]
	
	def swap(i, j):
		tmp = nums[i]
		nums[i] = nums[j]
		nums[j] = tmp
	
	while j < end:
		if nums[j] < pivot:
			i += 1
			swap(i, j)
			j += 1
		elif nums[j] >= pivot:
			j += 1
	i += 1
	swap(i, end)
	return i


def partition_sort(nums, start, end):
	sorted_point = partition(nums, start, end)
	
	if sorted_point - start > 1:
		partition_sort(nums, start, sorted_point - 1)
	if end - sorted_point > 1:
		partition_sort(nums, sorted_point + 1, end)


def testSingleNumber():
	print(singleNumber(None, [1, 2, 3, 4, 5, 6, 8, 6, 10, 11]))


def test_partition_sort():
	eddie = [12, 654, 3, 2, 2, 2, 1, 11, 223, 44, 5, 3, 167, 12, 111111, 2, 6, 87, 3, 2]
	# partition(eddie, 0, len(eddie) - 1)
	partition_sort(eddie, 0, len(eddie) - 1)
	print(eddie)


class Node:
	tree = ""
	right = ""
	left = ""
	
	def __init__(self, data):
		self.data = data


def build_minimal_tree(array):
	n = len(array)
	if len(array) == 1:
		return Node(array[0])
	if len(array) == 0:
		return None
	middle = n // 2
	root = Node(array[middle])
	root.left = build_minimal_tree(array[:middle])
	root.right = build_minimal_tree(array[middle + 1:])
	return root


def post_order(tree):
	print(tree.data)
	if tree.right:
		post_order(tree.right)
	if tree.left:
		post_order(tree.left)


SEPARATOR_AMOUNT = 100


def separate(amount=10):
	print("_" * amount)


def test_array_to_binary_tree():
	global tree
	array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
	tree = build_minimal_tree(array)
	post_order(tree)
	separate(SEPARATOR_AMOUNT)
	print(get_max_depth(tree))
	tmptree = Node(13)
	tmptree.left = Node(1)
	tmptree.left.left = Node(1)
	tmptree.left.left.left = Node(1)
	tmptree.left.left.left.left = Node(1)
	tmptree.left.left.left.left.left = Node(1)
	separate(SEPARATOR_AMOUNT)
	print(get_max_depth(tmptree))


def get_max_depth_helper(tree, depth, i):
	if tree.left == "" and tree.right == "":
		depth[0] = max(depth[0], i)
	elif tree.left:
		get_max_depth_helper(tree.left, depth, i + 1)
	elif tree.right:
		get_max_depth_helper(tree.right, depth, i + 1)


def get_max_depth(tree):
	depth = [0]
	i = 0
	get_max_depth_helper(tree, depth, i)
	return depth[0]


def topological_sort(graph):
	pass


def newMaxProfit(self, prices):
	"""
    :type prices: List[int]
    :rtype: int
    """
	current_sum = 0
	for i in range(1, len(prices) - 1):
		if prices[i + 1] - prices[i] > 0:
			current_sum += prices[i + 1] - prices[i]
		else:
			continue
	return current_sum


def rotate_arr(self, nums, k):
	"""
    :type nums: List[int]
    :type k: int
    :rtype: None Do not return anything, modify nums in-place instead.
    """
	j = 0
	i = 0
	n = len(nums)
	tmps = [nums[0], nums[k]]
	while i < len(nums):
		nums[(j + k) % n] = tmps[(i) % 2]
		j = (j + k) % n
		tmps[(i) % 2] = nums[(j + k) % n]
		i += 1
	return nums


def firstUniqChar(self, s):
	"""
    :type s: str
    :rtype: int
    """
	st = dict()
	for i in range(len(s)):
		if s[i] in st.keys():
			st[s[i]] = (st[s[i]][0] + 1, st[s[i]][1])
		else:
			st[s[i]] = (1, i)
	mini = -1
	for j in st:
		if st[j][0] == 1:
			if mini == -1:
				mini = st[j][1]
			else:
				mini = min(mini, st[j][1])
	return mini


def isValidBST(self, root):
	"""
    :type root: TreeNode
    :rtype: bool
    """
	if root is None:
		return True
	res = True
	if root.right:
		if root > root.right:
			return False
		else:
			res = isValidBST(self, root.right)
	if res:
		if root.left:
			if root < root.left:
				return False
			return isValidBST(self, root.right)
	return False


def levelOrderHelper(root, curLevel, levels):
	if root is None:
		return
	if len(levels) <= curLevel:
		levels.append([root.val])
	else:
		levels[curLevel].append(root.val)
	levelOrderHelper(root.right, curLevel + 1, levels)
	levelOrderHelper(root.left, curLevel + 1, levels)


def levelOrder(self, root):
	"""
    :type root: TreeNode
    :rtype: List[List[int]]
    """
	levels = []
	levelOrderHelper(root, 0, levels)
	return levels


def robogridify_helper(grid, i, j, new_arr):
	if i >= len(grid) or j >= len(grid[0]):
		return 0
	if i == len(grid) - 1 and j == len(grid[0]) - 1:
		return 1 if grid[i][j] == 0 else 0
	elif grid[i][j] == 1:
		return 0
	new_arr[i][j] = (robogridify_helper(grid, i + 1, j, new_arr) if new_arr[i + 1][j] == -1 else new_arr[i + 1][j]) + (
		robogridify_helper(grid, i, j + 1, new_arr) if new_arr[i][j + 1] == -1 else new_arr[i][j + 1])
	return new_arr[i][j]


def robogridify(grid):
	new_arr = [[-1] * (len(grid[0]) + 1)] * (len(grid) + 1)
	return robogridify_helper(grid, 0, 0, new_arr)


def robogridify2(grid):
	new_arr = [[-1] * (len(grid[0]) + 1) for _ in range(len(grid) + 1)]
	robogridify_helper(grid, 0, 0, new_arr)
	return new_arr[0][0]


def test_robogridify():
	grid = [[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
			[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
			[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
			[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
			[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
			[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
			[0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
			[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
			[0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
			[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
			[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
			[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
			[1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
			[1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]
	# import numpy as np
	# k = 11
	# l = 11
	# grid = np.ones((k,l))
	# print(grid)
	print(robogridify(grid))


def coinify_helper(n, arr, all_paths, current_path, amount):
	if n < 0:
		return
	if n == 0:
		amount[0] += 1
		all_paths.append(current_path[:])
		return amount[0]
	else:
		for i in arr:
			current_path.append(i)
			coinify_helper(n - i, arr, all_paths, current_path, amount)
			current_path.pop()
	return amount[0]


def coinify_memoize(n, arr, map):
	sm = 0
	if n < 0:
		return 0
	if n == 0:
		return 1
	else:
		for i in arr:
			if (n - i, i) in map.keys():
				sm += map[(n - i, i)]
			else:
				map[(n - i, i)] = coinify_memoize(n - i, arr, map)
				sm += map[(n - i, i)]
	return sm


# def coinify_memoize_unique(n, arr, map, all_paths, current_path):
# 	sm = 0
# 	if n < 0:
# 		return 0
# 	if n == 0:
# 		if current_path not in all_paths:
# 			new_set = current_path.copy()
# 			all_paths.add(frozenset(new_set))
# 			return 1
# 		return 0
# 	else:
# 		for i in arr:
# 			if (n - i, i) in map.keys():
# 				sm += map[(n - i, i)]
# 			else:
# 				current_path.add(i)
# 				map[(n - i, i)] = coinify_memoize_unique(n - i, arr, map, all_paths, current_path)
# 				current_path.remove(i)
# 				sm += map[(n - i, i)]
# 	return sm


def coinify(n, arr, backtrack=True, dynamic=True, unique=False, pall_paths=False):
	from datetime import datetime as d
	print(f"current i: {n}___________________________________")
	if backtrack:
		cur_time = d.now()
		current_path = []
		amount = [0]
		all_paths = []
		coinify_helper(n, arr, all_paths, current_path, amount)
		print(d.now() - cur_time)
		print(amount[0])
		if pall_paths:
			print(all_paths)
	if dynamic:
		cur_time = d.now()
		map = dict()
		print(d.now() - cur_time)
		print(coinify_memoize(n, arr, map))
	# if unique:
	# 	cur_time = d.now()
	# 	map = dict()
	# 	all_paths = set()
	# 	print(d.now() - cur_time)
	# 	current_path = set()
	# 	print(coinify_memoize_unique(n, arr, map, all_paths, current_path))
	# 	if pall_paths:
	# 		print(all_paths)
	print("___________________________________")
	return


def test_coinify_performance():
	# for i in range(0, 500, 2):
	# 	coinify(i, [25, 10, 5, 2, 1], backtrack=False, dynamic=True)
	coinify(3, [2, 1], backtrack=False, dynamic=True, unique=True, pall_paths=True)


def stack_boxes_helper_with_map(box_list, idx, current_height, min_w, min_h, min_d, map):
	found_height = 0
	for i in range(idx, len(box_list)):
		if box_list[i][0] < min_w and box_list[i][1] < min_h and box_list[i][2] < min_d:
			if (current_height + box_list[i][1], i + 1, True) in map.keys():
				with_cur = map[(current_height + box_list[i][1], i + 1, True)]
			else:
				with_cur = stack_boxes_helper_with_map(box_list, i + 1, current_height + box_list[i][1],
													   box_list[i][0],
													   box_list[i][1],
													   box_list[i][2], map)
				map[(current_height + box_list[i][1], i + 1, True)] = with_cur
			if (current_height, i + 1, False) in map.keys():
				without_cur = map[(current_height, i + 1, False)]
			else:
				without_cur = stack_boxes_helper_with_map(box_list, i + 1, current_height, min_w,
														  min_h,
														  min_d, map)
				map[(current_height, i + 1, False)] = without_cur
			found_height = max(found_height, without_cur, with_cur)
	if found_height == 0:
		return current_height
	else:
		return found_height


def stack_boxes_helper_no_map(box_list, idx, current_height, min_w, min_h, min_d):
	found_height = 0
	for i in range(idx, len(box_list)):
		if box_list[i][0] < min_w and box_list[i][1] < min_h and box_list[i][2] < min_d:
			result = stack_boxes_helper_no_map(box_list, idx + 1, current_height + box_list[i][1], box_list[i][0],
											   box_list[i][1],
											   box_list[i][2])
			found_height = max(found_height, result)
	if found_height == 0:
		return current_height
	else:
		return found_height


def stack_boxes():
	# idx 0 - width
	# idx 1 - height
	# idx 2 - depth
	# lst = [(2, 3, 2), (1, 1, 1), (6, 10, 6), (11, 11, 11), (10, 1, 2), (2, 2, 2), (5, 5, 5), (12, 12, 12), (12, 12,
	# 																										12)]
	import numpy as np
	res1, res2 = 0, 0
	lst = []
	h = 50
	while res1 == res2:
		from datetime import datetime as d
		print("___________________________________")
		lst = [(np.random.randint(1, h), np.random.randint(1, h), np.random.randint(1, h)) for i in range(100)]
		lst.sort(key=lambda x: x[2])  # doesn't matter how you sort it
		lst.reverse()
		print(lst)
		map = dict()
		cur_time = d.now()
		res1 = stack_boxes_helper_with_map(lst, 0, 0, float("inf"), float("inf"), float("inf"), map)
		print(d.now() - cur_time)
		cur_time = d.now()
		print(f"with map: {res1}")
		res2 = stack_boxes_helper_no_map(lst, 0, 0, float("inf"), float("inf"), float("inf"))
		print(d.now() - cur_time)
		print(f"no map: {res2}")


def test_boxes(n):
	# idx 0 - width
	# idx 1 - height
	# idx 2 - depth
	# lst = [(2, 3, 2), (1, 1, 1), (6, 10, 6), (11, 11, 11), (10, 1, 2), (2, 2, 2), (5, 5, 5), (12, 12, 12), (12,
	# 12,
	# 																										12)]
	import numpy as np
	res1, res2 = 0, 0
	lst = []
	h = 50
	import matplotlib.pyplot as plt
	map_diff_array = []
	no_map_diff_array = []
	from datetime import datetime as d
	for j in range(n):
		print(f"_____ {j} _____")
		lst = [(np.random.randint(1, h), np.random.randint(1, h), np.random.randint(1, h)) for i in range(j)]
		lst.sort(key=lambda x: x[2])  # doesn't matter how you sort it
		lst.reverse()
		map = dict()
		cur_time = d.now()
		res1 = stack_boxes_helper_with_map(lst, 0, 0, float("inf"), float("inf"), float("inf"), map)
		difference = d.now() - cur_time
		print(difference.total_seconds())
		if not map_diff_array:
			map_diff_array.append(difference.total_seconds())
		map_diff_array.append(max(difference.total_seconds(), max(map_diff_array)))
		print(f"with map: {res1}")
		cur_time = d.now()
		res2 = stack_boxes_helper_no_map(lst, 0, 0, float("inf"), float("inf"), float("inf"))
		difference = d.now() - cur_time
		print(difference.total_seconds())
		if not no_map_diff_array:
			no_map_diff_array.append(difference.total_seconds())
		no_map_diff_array.append(max(difference.total_seconds(), max(no_map_diff_array)))
		print(f"with map: {res2}")
		if j % 10 == 0:
			plt.plot(map_diff_array)
			plt.title(f"sizes passed: {j}")
			plt.plot(no_map_diff_array)
			plt.show()
	plt.plot(map_diff_array)
	plt.plot(no_map_diff_array)
	plt.show()


def boxes(n=100):
	# print(stack_boxes())
	test_boxes(n)


def robogridify_dual_prints():
	grid = [["8", "3", ".", ".", "7", ".", ".", ".", "."]
		, ["6", ".", ".", "1", "9", "5", ".", ".", "."]
		, [".", "9", "8", ".", ".", ".", ".", "6", "."]
		, ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
		, ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
		, ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
		, [".", "6", ".", ".", ".", ".", "2", "8", "."]
		, [".", ".", ".", "4", "1", "9", ".", ".", "5"]
		, [".", ".", ".", ".", "8", ".", ".", "7", "9"]]
	print(robogridify2(grid))
	print(robogridify(grid))


def isValidSudoku(self, board):
	"""
    :type board: List[List[str]]
    :rtype: bool
    """
	n = len(board)
	shoresh = int(sqrt(n))
	for i in range(n):
		for j in range(n):
			posi = (i // shoresh) * shoresh
			posj = (j // shoresh) * shoresh
			f = board[i][j]
			if board[i][j] == ".":
				continue
			for k in range(n):
				if board[i][j] == board[i][k] and k != j:
					return False
				if board[i][j] == board[k][j] and i != k:
					r = board[k][j]
					return False
			for k in range(shoresh):
				for l in range(shoresh):
					if l == j % shoresh and k == i % shoresh:
						continue
					if board[posi + k][posj + l] == board[i][j]:
						return False
	return True


def binary_helper(element, array, min, max):
	middle = (min + max) // 2
	if min == max:
		return max  # if we got here, this is the closest element to element
	if array[middle] != element:
		if array[middle] < element:
			return binary_helper(element, array, middle + 1, max)
		if array[middle] > element:
			return binary_helper(element, array, 0, middle)
	if array[middle] == element:
		return middle


def bsearch(element, array):
	return binary_helper(element, array, 0, len(array))


def findTheDistanceValue(self, arr1, arr2, d):
	"""
    :type arr1: List[int]
    :type arr2: List[int]
    :type d: int
    :rtype: int
    """
	arr1.sort()
	arr2.sort()
	counter = len(arr1)
	for k in arr1:
		idx = bsearch(k, arr2)
		if idx < len(arr2):
			if abs(arr2[idx] - k) <= d:
				counter -= 1
				continue
		if idx + 1 < len(arr2):
			if abs(arr2[idx + 1] - k) <= d:
				counter -= 1
				continue
		if idx - 1 >= 0:
			if abs(arr2[idx - 1] - k) <= d:
				counter -= 1
				continue
	
	return counter


def lenLongestFibSubseq(self, arr):
	"""
    :type arr: List[int]
    :rtype: int
    """
	mp = dict()
	for i in range(len(arr)):
		mp[arr[i]] = (1, None)
	i = 0
	while i < len(arr) - 1:
		j = i + 1
		while j < len(arr):
			if arr[j] - arr[i] in mp.keys():
				if mp[arr[j] - arr[i]][1] == arr[j] or mp[arr[j] - arr[i]][1] is None:
					mp[arr[j]] = (mp[arr[j] - arr[i]][0] + 1, arr[j]) if mp[arr[j] - arr[i]][0] + 1 > mp[arr[j]][
						0] else (mp[arr[j]][0], mp[arr[j]][1])
			j += 1
		i += 1
	maxi = 0
	for k, l in mp.values():
		maxi = max(maxi, k)
	return maxi


def combinationSum4(self, nums, target):
	"""
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
	mp = dict()
	# return combHelper(nums, target, mp)
	dp = [0] * (target + 1)
	return combHelper2(nums, target + 1, dp)


def combHelper(nums, target, mp):
	if target == 0:
		mp[target] = 1
		return 1
	elif target < 0:
		return 0
	else:
		cur_ways = 0
		for i in nums:
			if target - i in mp:
				cur_ways += mp[target - i]
			else:
				cur_ways += combHelper(nums, target - i, mp)
		mp[target] = cur_ways
		return cur_ways


def combHelper2(nums, target, dp):
	for j in nums:
		if j < len(dp):
			dp[j] = 1
	for i in range(target):
		for j in range(len(nums)):
			if i - nums[j] >= 0:
				dp[i] += dp[i - nums[j]]
	return dp[-1]


def combinationSum(self, candidates, target):
	cur_path = []
	unique_paths = []
	return combSumHelper(candidates, target, cur_path, 0, unique_paths)


def combSumHelper(candidates, target, cur_path, idx, unique_paths):
	"""
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
	if target == 0:
		unique_paths.append(cur_path[:])
		return
	if idx >= len(candidates) or target < 0:
		return
	i = 0
	curRes = 0
	while target - curRes >= 0:
		combSumHelper(candidates, target - curRes, cur_path, idx + 1, unique_paths)
		cur_path.append(candidates[idx])
		curRes += candidates[idx]
		i += 1
	while i > 0:
		cur_path.pop()
		i -= 1
	return unique_paths


def canPartition(self, nums):
	"""
    :type nums: List[int]
    :rtype: bool
    """
	if nums is None:
		return False
	sumA = 0
	sumB = sum(nums)
	if sumA == sumB:
		return True
	for i in range(len(nums)):
		for j in range(i + 1, len(nums)):
			sumA += nums[j]
			sumB -= nums[j]
			if sumA == sumB:
				return True
			sumA -= nums[j]
			sumB += nums[j]
		sumA += nums[i]
		sumB -= nums[i]
		if sumA == sumB:
			return True
	if sumA == sumB:
		return True
	return False


def numberOfArithmeticSlices(self, nums):
	"""
    :type nums: List[int]
    :rtype: int
    """
	dp = [[0, 0]] * len(nums)
	i = len(nums) - 3
	current_sum = 0
	while i >= 0:
		if nums[i + 2] - nums[i + 1] == nums[i + 1] - nums[i]:
			if dp[i + 1][0] == 0:
				dp[i] = (1, 2)
			else:
				dp[i] = [dp[i + 1][0] + dp[i + 1][1], dp[i + 1][1] + 1]
		else:
			current_sum += dp[i + 1][0]
		i -= 1
	current_sum += dp[0][0]
	return current_sum


def uniquePaths(self, m, n):
	dp = [[0] * (m + 2) for _ in range(n + 2)]
	return uniquePathsHelper(m, n, 0, 0, dp)


def uniquePathsHelper(m, n, i, j, dp):
	if i < 0 or i > n:
		return 0
	elif j < 0 or j > m:
		return 0
	elif i == n - 1 and j == m - 1:
		return 1
	else:
		if dp[i + 1][j] == 0:
			res1 = uniquePathsHelper(m, n, i + 1, j, dp)
		else:
			res1 = dp[i + 1][j]
		if dp[i][j + 1] == 0:
			res2 = uniquePathsHelper(m, n, i, j + 1, dp)
		else:
			res2 = dp[i][j + 1]
		dp[i][j] = res1 + res2
		return res1 + res2


def find_all_palindromes(self, s):
	"""
        :type s: str
        :rtype: List[List[str]]
        """
	all_partitions = []
	cur_partition = []
	all_palindromes_helper(s, 0, cur_partition, all_partitions)
	return all_partitions


def all_palindromes_helper(s, start, cur_partition, all_partitions):
	if start == len(s):
		all_partitions.append(cur_partition[:])
		return
	d = 0
	for i in range(start, len(s)):
		if is_palindrome(s[start:i + 1]):
			cur_partition.append(s[start:i + 1])
			all_palindromes_helper(s, i + 1, cur_partition, all_partitions)
			cur_partition.pop()


def is_palindrome(s):
	for i in range(len(s) // 2):
		if s[i] != s[-i - 1]:
			return False
	return True


def checkPartitioning(self, s):
	"""
        :type s: str
        :rtype: List[List[str]]
        """
	all_partitions = []
	cur_partition = []
	mp = dict()
	return p_3_helper(s, 0, cur_partition, all_partitions, 0, mp)


def p_3_helper(s, start, cur_partition, all_partitions, d, mp):
	if d > 3:
		return False
	if is_palindrome(s[start:]) and d == 2:
		return True
	if start == len(s) and d == 3:
		return True
	for i in range(start, len(s)):
		if is_palindrome(s[start:i + 1]):
			cur_partition.append(s[start:i + 1])
			res = p_3_helper(s, i + 1, cur_partition, all_partitions, d + 1, mp)
			if res == True:
				return True
			mp[s[start:i + 1]] = False
			cur_partition.pop()
	return False


def numDecodings(self, s):
	"""
    :type s: str
    :rtype: int
    """
	if len(s) == 1:
		if int(s[0]) == 0:
			return 0
		else:
			return 1
	dp = [0] * (len(s) + 2)
	dp[-1] = 0
	dp[-2] = 1
	cameFromZero = False
	i = len(s) - 1
	while i >= 0:
		if int(s[i]) == 0:
			if i - 1 >= 0 and int(s[i - 1]) in {1, 2}:
				dp[i] = dp[i + 1]
				dp[i - 1] = dp[i + 1]
				i -= 2
				cameFromZero = True
				continue
			else:
				return 0
		elif int(s[i]) == 2:
			if i + 1 < len(s) and int(s[i + 1]) <= 6:
				dp[i] = dp[i + 1] + dp[i + 2] if not cameFromZero else dp[i + 1]
			else:
				dp[i] = dp[i + 1]
		elif int(s[i]) == 1:
			dp[i] = dp[i + 1] + dp[i + 2] if not cameFromZero else dp[i + 1]
		else:
			dp[i] = dp[i + 1]
		cameFromZero = False
		i -= 1
	return dp[0]


# def cherryHelper(grid, i1, j1, i2, j2, dp):
# 	if i1 < 0 or i2 < 0 or i1 >= len(grid) or i2 >= len(grid) or j1 >= len(grid[0]) or j2 >= len(
# 			grid[0]) or j1 < 0 or j2 < 0:
# 		return 0
# 	else:
# 		cur_score = 0
# 		if j1 == j2:
# 			cur_score += grid[i1][j1]
# 		else:
# 			cur_score += grid[i1][j1] + grid[i2][j2]
#
#
# 		max1 = cherryHelper(grid, i1 + 1, j1, i2 + 1, j2, dp) if (j1,j2,i1+1) not in dp else dp[(j1,j2, i1+1)]
# 		max2 = cherryHelper(grid, i1 + 1, j1, i2 + 1, j2 + 1, dp) if (j1,j2+1, i1+1) not in dp else dp[(j1,j2+1,
# 		i1+1)]
# 		max3 = cherryHelper(grid, i1 + 1, j1, i2 + 1, j2 - 1, dp) if (j1,j2-1, i1+1) not in dp else dp[(j1,j2-1,
# 		i1+1)]
# 		max4 = cherryHelper(grid, i1 + 1, j1 + 1, i2 + 1, j2, dp) if (j1+1,j2, i1+1) not in dp else dp[(j1+1,j2,
# 		i1+1)]
# 		max5 = cherryHelper(grid, i1 + 1, j1 + 1, i2 + 1, j2 + 1, dp) if (j1+1,j2+1, i1+1) not in dp else dp[(j1+1,
# 		j2+1, i1+1)]
# 		max6 = cherryHelper(grid, i1 + 1, j1 + 1, i2 + 1, j2 - 1, dp) if (j1+1,j2-1, i1+1) not in dp else dp[(j1+1,
# 		j2-1, i1+1)]
# 		max8 = cherryHelper(grid, i1 + 1, j1 - 1, i2 + 1, j2, dp) if (j1-1,j2, i1+1) not in dp else dp[(j1-1,j2,
# 		i1+1)]
# 		max7 = cherryHelper(grid, i1 + 1, j1 - 1, i2 + 1, j2 + 1, dp) if (j1-1,j2+1, i1+1) not in dp else dp[(j1-1,
# 		j2+1, i1+1)]
# 		max9 = cherryHelper(grid, i1 + 1, j1 - 1, i2 + 1, j2 - 1, dp) if (j1-1,j2-1, i1+1) not in dp else dp[(j1-1,
# 		j2-1, i1+1)]
# 		found_maxi = max(max1, max2, max3, max4, max5, max6, max7, max8, max9)
# 		max_score = cur_score + found_maxi
# 		dp[(j1, j2, i1)] = found_maxi + cur_score
# 		return max_score


def cherryHelper(grid, i1, j1, i2, j2, dp):
	if i1 < 0 or i2 < 0 or i1 >= len(grid) or i2 >= len(grid) or j1 >= len(grid[0]) or j2 >= len(
			grid[0]) or j1 < 0 or j2 < 0:
		return 0
	else:
		if j1 == j2:
			cur_score = grid[i1][j1]
		else:
			cur_score = grid[i1][j1] + grid[i2][j2]
		dp[(j1, j2, i1)] = cur_score + \
						   max(cherryHelper(grid, i1 + 1, j1 + i, i2 + 1, j2 + j, dp)
							   if (j1 + i, j2 + j, i1 + 1) not in dp else dp[(j1 + i, j2 + j, i1 + 1)]
							   for i in [-1, 0, 1] for j in [-1, 0, 1])
		return dp[(j1, j2, i1)]


def cherryPickupRecursive(self, grid):
	"""
    :type grid: List[List[int]]
    :rtype: int
    """
	dp = dict()
	return cherryHelper(grid, 0, 0, 0, len(grid[0]) - 1, dp)


def cherryPickupIterative(self, grid):
	"""
    :type grid: List[List[int]]
    :rtype: int
    """
	dp = dict()
	return cherryHelper(grid, 0, 0, 0, len(grid[0]) - 1, dp)


def longestPalindrome(self, s):
	"""
    :type s: str
    :rtype: str
    """
	palin = ""
	for i in range(len(s)):
		for j in range(i, len(s)):
			if len(palin) <= j - i:
				if is_palindrome(s[i:j + 1]):
					palin = s[i:j + 1]
	return palin


def minPathSum(self, grid):
	"""
    :type grid: List[List[int]]
    :rtype: int
    """
	for i in range(len(grid) - 1, -1, -1):
		for j in range(len(grid[0]) - 1, -1, -1):
			if i + 1 >= len(grid) and j + 1 >= len(grid[0]):
				continue
			if j + 1 >= len(grid[0]):
				grid[i][j] = grid[i][j] + grid[i + 1][j]
			elif i + 1 >= len(grid):
				grid[i][j] = grid[i][j] + grid[i][j + 1]
			elif j + 1 < len(grid[0]) and i + 1 < len(grid):
				grid[i][j] = grid[i][j] + min(grid[i + 1][j], grid[i][j + 1])
	return grid[0][0]


def tests():
	test_kadanes()
	test_queue()
	print(binary_sort([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0]))
	test_count_island()
	test_find_max_subseq()
	print(climbStairs(10))
	test_max_price()
	test_rob()
	test_merge()
	print(firstBadVersion(None, 900))
	print(removeDuplicates(None, [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))
	testSingleNumber()
	test_partition_sort()
	test_array_to_binary_tree()
	graph = Node
	print(topological_sort(graph))
	print(newMaxProfit(None, [7, 1, 5, 3, 6, 4]))
	print(rotate_arr(None, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1))
	print(firstUniqChar(None, "sshhit"))
	test_robogridify()
	robogridify_dual_prints()
	test_coinify_performance()
	boxes(250)
	grid = [["5", "3", ".", ".", "7", ".", ".", ".", "."]
		, ["6", ".", ".", "1", "9", "5", ".", ".", "."]
		, [".", "9", ".", ".", ".", ".", ".", "6", "."]
		, ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
		, ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
		, ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
		, [".", "6", ".", ".", ".", ".", "2", "8", "."]
		, [".", ".", ".", "4", "1", "9", ".", ".", "5"]
		, [".", ".", ".", ".", "8", ".", ".", "7", "9"]]
	print(isValidSudoku(None, grid))
	print(bsearch(10.1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
	a = [-3, -3, 4, -1, -10]
	b = [7, 10]
	print(findTheDistanceValue(None, a, b, 12))
	print(combinationSum4(None, [1, 2, 3], 4))
	print(combinationSum(None, sorted([13, 29, 31, 40, 51]), 147))
	print(canPartition(None, [1, 5, 11, 5]))
	print(canPartition(None, [1, 5, 5, 1]))
	kaki = [14, 9, 8, 4, 3, 2]
	print(numberOfArithmeticSlices(None, [1, 2, 3, 4, 5, 0, 3, 6, 9, 12]))
	print(uniquePaths(None, 12, 12))
	print(is_palindrome("abcddcbaa"))
	print(find_all_palindromes(None, "aaaaaaaaaaaaaabcdedcbaaaaaaaaaaaaaa"))
	pipi = \
		"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	print(checkPartitioning(None, pipi))
	print(numDecodings(None, "2101"))
	cur = [[1, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 3, 0], [2, 0, 9, 0, 0, 0, 0], [0, 3, 0, 5, 4, 0, 0],
		   [1, 0, 2, 3, 0, 0, 6]]
	x = 100
	# cur = [[1] * x for _ in range(x)]
	cur = [[3, 1, 1], [2, 5, 1], [1, 5, 5], [2, 1, 1]]
	print(cherryPickupRecursive(None, cur))
	grid = [[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
			[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
			[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
			[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
			[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
			[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
			[0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
			[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
			[0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
			[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
			[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
			[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
			[1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
			[1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]
	print(robogridify2(grid))
	tst = \
		"esbtzjaaijqkgmtaajpsdfiqtvxsgfvijpxrvxgfumsuprzlyvhclgkhccmcnquukivlpnjlfteljvykbddtrpmxzcrdqinsnlsteonhcegtkoszzonkwjevlasgjlcquzuhdmmkhfniozhuphcfkeobturbuoefhmtgcvhlsezvkpgfebbdbhiuwdcftenihseorykdguoqotqyscwymtjejpdzqepjkadtftzwebxwyuqwyeegwxhroaaymusddwnjkvsvrwwsmolmidoybsotaqufhepinkkxicvzrgbgsarmizugbvtzfxghkhthzpuetufqvigmyhmlsgfaaqmmlblxbqxpluhaawqkdluwfirfngbhdkjjyfsxglsnakskcbsyafqpwmwmoxjwlhjduayqyzmpkmrjhbqyhongfdxmuwaqgjkcpatgbrqdllbzodnrifvhcfvgbixbwywanivsdjnbrgskyifgvksadvgzzzuogzcukskjxbohofdimkmyqypyuexypwnjlrfpbtkqyngvxjcwvngmilgwbpcsseoywetatfjijsbcekaixvqreelnlmdonknmxerjjhvmqiztsgjkijjtcyetuygqgsikxctvpxrqtuhxreidhwcklkkjayvqdzqqapgdqaapefzjfngdvjsiiivnkfimqkkucltgavwlakcfyhnpgmqxgfyjziliyqhugphhjtlllgtlcsibfdktzhcfuallqlonbsgyyvvyarvaxmchtyrtkgekkmhejwvsuumhcfcyncgeqtltfmhtlsfswaqpmwpjwgvksvazhwyrzwhyjjdbphhjcmurdcgtbvpkhbkpirhysrpcrntetacyfvgjivhaxgpqhbjahruuejdmaghoaquhiafjqaionbrjbjksxaezosxqmncejjptcksnoq"
	print(len(tst))
	print(longestPalindrome(None,
							tst))


def splitArrayHelperGetAllResults(nums, start, m, cur_arrays, all_results, best, mp):
	if m == 0 and start == len(nums):
		all_results.append(cur_arrays[:])
		for k in cur_arrays:
			summed = sum(k)
			if best[0] < summed:
				best[0] = summed
				best[1] = k[:]
	elif m == 0:
		return
	tmp = []
	for i in range(start, len(nums)):
		tmp.append(nums[i])
		cur_arrays.append(tmp[:])
		splitArrayHelperGetAllResults(nums, i + 1, m - 1, cur_arrays, all_results, best, mp)
		cur_arrays.pop()


def splitArrayHelper(nums, start, m, cur_arrays, mp):
	if m == 1:
		return sum(nums[start:])
	elif start == len(nums):
		return 0
	if (start, m) in mp:
		return mp[start, m]
	else:
		mp[start, m] = float("inf")
		for i in range(start, len(nums)):
			l, r = sum(nums[start:i]), splitArrayHelper(nums, i, m - 1, cur_arrays, mp)
			mp[start, m] = min(mp[start, m], max(l, r))
	return mp[start, m]


def splitArray(self, nums, m):
	"""
    :type nums: List[int]
    :type m: int
    :rtype: int
    """
	cur_arrays = []
	all_results = []
	best = [0, []]
	# splitArrayHelperGetAllResults(nums, 0, m, cur_arrays, all_results, best, mp)
	# print(best[0])
	# print(best[1])
	mp = dict()
	rev_nums = nums[::-1]
	cur_arrays = []
	res2 = splitArrayHelper(nums, 0, m, cur_arrays, mp)
	cur_arrays = []
	mp = dict()
	res1 = splitArrayHelper(rev_nums, 0, m, cur_arrays, mp)
	mn = min(res1, res2)
	print(mn)
	cache = defaultdict(dict)
	print(another_helper(None, 0, nums, m, cache))
	return best[0]


def another_helper(self, i, nums, m, cache):
	if i == len(nums):
		return 0
	elif m == 1:
		return sum(nums[i:])
	else:
		if i in cache and m in cache[i]:
			return cache[i][m]
		cache[i][m] = float('inf')
		for j in range(1, len(nums) + 1):
			left, right = sum(nums[i:i + j]), another_helper(None, i + j, nums, m - 1, cache)
			cache[i][m] = min(cache[i][m], max(left, right))
			if left > right:
				break
		return cache[i][m]


class ListNode(object):
	def __init__(self, val=0, next=None):
		self.val = val
		self.next = next


def sortList(self, head):
	"""
    :type head: ListNode
    :rtype: ListNode
    """
	from sys import setrecursionlimit
	setrecursionlimit(200000)
	sortHelper(head, None)
	return head


def sortHelper(head, end):
	if head is None or head.next is None:
		return head
	if head == end:
		return head
	
	def make_nlogn(head, end):
		import numpy as np
		cur = head
		size = 0
		while cur != end:
			cur = cur.next
			size += 1
		size //= np.random.randint(2, 4)
		if size > 0:
			cur = head
			for i in range(size):
				cur = cur.next
			tmp = head.val
			head.val = cur.val
			cur.val = tmp
	
	make_nlogn(head, end)
	j = head.next
	i = head
	pivot = head
	while j != end:
		if j.val <= pivot.val:
			i = i.next
			tmp = i.val
			i.val = j.val
			j.val = tmp
		j = j.next
	tmp = pivot.val
	pivot.val = i.val
	i.val = tmp
	sortHelper(head, i)
	sortHelper(i.next, end)
	return head


def printLinkedList(head):
	cur = head
	while cur != None:
		print(f"{cur.val}->", end="")
		cur = cur.next
	print()


def read_from_lst(lst):
	cur = ListNode(lst[0])
	head = cur
	for i in lst[1:]:
		cur.next = ListNode(i)
		cur = cur.next
	return head


def assertSort(head):
	prev = head
	cur = head.next
	i = 0
	while cur != None:
		if not (cur.val >= prev.val):
			print(cur.val)
		cur = cur.next
		prev = prev.next
		i += 1


# print(i)


if __name__ == '__main__':
	# grid = [[6, 1, 2], [1, 3, 6], [7, 2, 5]]
	# print(minPathSum(None, grid))
	# arr = []
	# import numpy as np
	#
	# np.random.seed(0)
	# for s in range(100):
	#     arr.append(np.random.randint(0, 120))
	# # arr = [1000, 5000, 6, 7]
	# splitArray(None, arr, 10)
	# head = getHead2()
	import numpy as np
	
	arr = [(i % 3) + 1 for i in range(10000, 0, -1)]
	head = read_from_lst(arr)
	printLinkedList(head)
	sortList(None, head)
	printLinkedList(head)
	assertSort(head)
