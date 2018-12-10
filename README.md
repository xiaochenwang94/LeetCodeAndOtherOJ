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

