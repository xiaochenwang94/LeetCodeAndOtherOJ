# LeetCodeAndOtherOJ

## 背包问题
*0-1背包问题*

leetcode 416 Partition Equal Subset Sum 

问题描述：
给定一个正整数的numlist，划分成加和相等的两部分。

思路：
可以转换为0/1背包问题，加和除以2就是背包的容量。这道题要求背包恰好装满。

    class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        s = sum(nums)
        V = s // 2
        if 2*V != s:
            return False
        MIN_INF = -20000 * 101
        l = [MIN_INF] * (V+1)
        l[0] = 0
        for i in range(1, len(nums)+1):
            for v in range(V, nums[i-1]-1, -1):
                l[v] = max(l[v], l[v-nums[i-1]]+1)
        if l[V] > 0 :
            return True
        else:
            return False

## 其他动态规划问题

leetcode 53 Maximum Subarray

问题描述：给定一个数组，寻找最大的自数组。

思路：e = max(a_i, e)，e代表之前子序列的最大值。

    class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        e = nums[0]
        max_e = e
        for i in range(1, len(nums)):
            e = max(e+nums[i], nums[i])
            max_e = max(e, max_e)
        return max_e


## 按顺序刷题

leetcode 11 Container With Most Water    

问题描述：找出能装的最多水的两个柱子。

思路：开始想复杂了，设置两个指针，一个自左向右，另一个自右向左。如果height[i] < height[j] i++ else j--; 一个小优化，当移动一个柱子的时候，可以连续移动到一个比当前大的柱子。

    class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if len(height) == 0:
            return 0
        i, j = 0, len(height)-1
        ma = 0
        while i < j:
            ma = max(ma, (j-i)*min(height[i], height[j]))
            if height[i] < height[j]:
                th = height[i]
                while(height[i]<=th and i<j):
                    i+=1
            else:
                th = height[j]
                while(height[j]<=th and i<j):
                    j-=1
        return ma

leetcode 12 Integer to Roman  

问题描述：整数转换为罗马数字

思路：和进制转换一样，把特殊的情况（4, 9, 40, 90, 400, 900)考虑进去就行了。


    class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        d = {
            1: 'I', 4: 'IV', 5: 'V', 9: 'IX', 
            10: 'X', 40: 'XL', 50: 'L', 90: 'XC',
            100: 'C', 400: 'CD', 500: 'D', 900: 'CM', 1000:'M'
        }
        l = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        i=0
        ret = ""
        for x in l:
            ret+=d[x]*(num//x)
            num %= x
        return ret

leetcode 13 Roman to Integer   

问题描述：罗马数字转换为整数

思路: 和上一题相反，一个规律就是特殊情况不会出现在正常情况前面，例如：V总是出现在IV前，出现了IV之后就不可能出现V。顺序扫描相加即可。


    class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        d = {
            'C': 100, 'CD': 400, 'CM': 900, 'D': 500, 'I': 1,
            'IV': 4, 'IX': 9, 'L': 50, 'M': 1000, 'V': 5, 'X': 10,
            'XC': 90, 'XL': 40
        }
        l = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        ret = 0
        idx = 0
        for x in l:
            if idx >= len(s):
                break
            while idx< len(s) and  s[idx] == x:
                ret+=d[x]
                idx+=1
            while idx < len(s)-1 and s[idx:idx+2] == x:
                ret+=d[x]
                idx+=2
        return ret
                
leetcode 14 Longest Common Prefix

问题描述：给定一个字符串的list，找出最长的前缀。例如：
    
    Input: ["flower","flow","flight"]
    Output: "fl"
思路：直接便利寻找即可。这道题还有一种巧妙的解法，给字符串排序，之后只需要比较第一个和最后一个字符串即可。

    class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        size = len(strs)
        if size == 0:
            return ""
        idx = 0
        flag = True
        shortest = min([len(x) for x in strs])
        while flag:
            if idx == shortest:
                break
            c = strs[0][idx]
            for s in strs:
                if s[idx] != c:
                    flag = False
                    break
            idx+=1
        if not flag:
            idx = max(0, idx-1)
        return strs[0][:idx]


leetcode 15 3Sum

问题描述：找出数组中三个相加为0的数字，返回所有结果。

思路：这道题参考了答案，弄个半个小时才通过。开始使用暴力求解，果断超时。解法中先把列表排序，然后从左到右扫一次，每次固定一个数字，找另外两个数字的和与当前数字相反。当扫到正数就可以结束了，因子正数后面都是正数，不可能找到结果。

在找后面两个数字的时候，左边的数字时i=idx+1，右边的是j=size-1。当i < j的时候寻找。如果找到了两个数字，那么添加到结果中，之后进行剪枝，当nums[i] == nums[i+1], nums[j] == nums[j-1]，就过滤掉，不这样求出的结果会重复。

找完后面两个数字之后，同样的固定的数字也进行剪枝。

    class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        size = len(nums)
        if size <3:
            return []
        nums = sorted(nums)
        idx = 0 
        ret = []
        while idx < size and nums[idx] <= 0:
            target = 0 - nums[idx]
            i = idx+1
            j = size-1
            while i<j:
                if nums[i] + nums[j] < target:
                    i+=1
                elif nums[i] + nums[j] > target:
                    j-=1
                else:
                    ret.append([nums[idx], nums[i], nums[j]])
                    while i < j and nums[i] == nums[i+1]:
                        i+=1
                    while i < j and nums[j] == nums[j-1]:
                        j-=1
                    i+=1
                    j-=1
            while idx < size-1 and nums[idx] == nums[idx+1]:
                idx+=1
            idx+=1
        return ret

leetcode 16 3Sum Closest

问题描述: 和上一道题相似，现在要求加和和给定的target最接近。

思路：和上一题一样，固定一个数字，找另外两个数字。使用两个index从左到右，从右到左，比三层循环次数要少，可以加速。其他剪枝方法不太好用。

    class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums = sorted(nums)
        idx = 0
        size = len(nums)
        close = 0
        diff = 1e10
        while idx < size:
            i = idx+1
            j = size-1
            while i < j:
                s = nums[idx] + nums[i] + nums[j]
                if  s < target:
                    i+=1
                    if diff > target - s:
                        close = s
                        diff = target - s
                elif s > target:
                    j-=1
                    if diff > s - target:
                        close = s
                        diff = s - target
                else:
                    return target
            while idx < size-1 and nums[idx] == nums[idx+1]:
                idx+=1
            idx+=1
        return close


            