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


leetcode 18 4Sum

问题描述：四个数字加和

思路：和前一道题目一样，套一层for循环就行了。

    class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        nums = sorted(nums)
        size = len(nums)
        ret = []
        for fidx in range(len(nums)-3):
            if fidx!=0:
                if nums[fidx] == nums[fidx-1]:
                    continue
            t = target - nums[fidx]
            idx = fidx+1
            while idx < size-2:
                tt = t - nums[idx]
                i = idx + 1
                j = size - 1
                while i < j:
                    s = nums[i] + nums[j]
                    if s < tt:
                        i+=1
                    elif s > tt:
                        j-=1
                    else:
                        ret.append([nums[fidx], nums[idx], nums[i], nums[j]])
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
                    
                    
leetcode 19 Remove Nth Node From End of List

问题描述：删除倒数第n个数字

思路：两个指针，第一个先跑n个，第二个再跑。然后删除。

    # Definition for singly-linked list.
    # class ListNode(object):
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None

    class Solution(object):
        def removeNthFromEnd(self, head, n):
            """
            :type head: ListNode
            :type n: int
            :rtype: ListNode
            """
            p = head
            q = head
            for i in range(n):
                p = p.next
            if p is None:
                return head.next
            while p.next is not None:
                p = p.next
                q = q.next
            q.next = q.next.next
            return head

leetcode 21 Merge Two Sorted Lists

问题描述：合并两个有序的链表

思路：参考了大神的代码，递归写法。每次比较l1, l2如果l1.val > l2.val 那么交换两段列表，然后l1向后走一个。

    # Definition for singly-linked list.
    # class ListNode(object):
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None

    class Solution(object):
        def mergeTwoLists(self, l1, l2):
            """
            :type l1: ListNode
            :type l2: ListNode
            :rtype: ListNode
            """
            if not l1 or (l2 and l1.val > l2.val):
                t = l1
                l1 = l2
                l2 = t
            if l1:
                l1.next = self.mergeTwoLists(l1.next, l2)
            return l1


leetcode 22 Generate Parentheses

问题描述：给定n对括号，求出所有可能。

思路：

解法一：BFS进行搜索，每次分两种可能，添加左括号或者右括号。

    class Node(object):
    def __init__(self, n):
        self.left = n
        self.right = n
        self.stack = 0
        self.val = ""

    class Solution(object):
        def generateParenthesis(self, n):
            """
            :type n: int
            :rtype: List[str]
            """
            import Queue
            q = Queue.Queue()
            ret = []
            q.put(Node(n))
            while not q.empty():
                node = q.get()
                if node.left == 0 and node.right == 0:
                    ret.append(node.val)
                    continue
                if node.left != 0:
                    n = Node(n)
                    n.left = node.left - 1
                    n.right = node.right
                    n.stack = node.stack+1
                    n.val = node.val+'('
                    q.put(n)
                if node.right != 0 and node.stack > 0:
                    n = Node(n)
                    n.left = node.left
                    n.right = node.right - 1
                    n.stack = node.stack - 1
                    n.val = node.val+')'
                    q.put(n)
            return ret

解法2: dfs

    class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        ret = []
        self.dfs(n, n, "", ret)
        return ret
    
    def dfs(self, left, right, val, ret):
        if left == 0 and right == 0:
            ret.append(val)
        if right < left:
            return None
        if left != 0:
            self.dfs(left-1, right, val+"(", ret)
        if right != 0 and right-1>=left:
            self.dfs(left, right-1, val+")", ret)

leetcode 29 Divide Two Integers

问题描述：给定两个正数，求整除之后的结果。不能用乘法，除法，mod。

思路：使用递归模拟，当除数大于被除数直接返回，否则加上初始被除数的倍数。**这道题应该注意题目描述的边界条件，以及两个数符号相反的情况！**

    class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        if (dividend >= 0 and divisor > 0) or (dividend < 0 and divisor < 0):
            flag = 1
        else:
            flag = -1
        dividend = abs(dividend)
        divisor = abs(divisor)
        l = [1]
        r = self.dfs(dividend, divisor, l)
        r = r[0]*flag
        if r > 2147483647:
            return 2147483647
        if r < -2147483648:
            return -2147483648
        return r
    
    def dfs(self, dividend, divisor, l):
        if dividend >= divisor:
            l.append(l[-1]+l[-1])
            num, d = self.dfs(dividend, divisor+divisor, l)
            l.pop()
            if d >= divisor: 
                d -= divisor
                num += l[-1]
            return num, d
        return 0, dividend

leetcode 100 Same Tree

问题描述：给定两棵树，判断是不是相等的两棵树。

思路：开始想按序遍历，后来发现需要在遍历的时候考虑null的问题。要不然不同的两棵树遍历的结果可能相同。例如：[1,1],[1,null,1]。可以同时遍历两棵树，只要结果不同就可以返回False了。

    # Definition for a binary tree node.
    # class TreeNode(object):
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None

    class Solution(object):
        def isSameTree(self, p, q):
            """
            :type p: TreeNode
            :type q: TreeNode
            :rtype: bool
            """
            return self.dfs(p, q)
        
        def dfs(self, root1, root2):
            if root1 is None and root2 is None:
                return True
            if root1 is None or root2 is None:
                return False
            if root1.val != root2.val:
                return False
            return self.dfs(root1.left, root2.left) and self.dfs(root1.right, root2.right)
            
## Leetcode Contest
contest 116 大神代码

**961. N-Repeated Element in Size 2N Array**

In a array A of size 2N, there are N+1 unique elements, and exactly one of these elements is repeated N times.

Return the element repeated N times.

Input: [1,2,3,3]
Output: 3
```cpp
#include<map>
#include<stdio.h>
#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include<string.h>
using namespace std;

typedef long long LL;
typedef vector<int> VI;

#define REP(i,n) for(int i=0, i##_len=(n); i<i##_len; ++i)
#define EACH(i,c) for(__typeof((c).begin()) i=(c).begin(),i##_end=(c).end();i!=i##_end;++i)
#define eprintf(...) fprintf(stderr, __VA_ARGS__)

template<class T> inline void amin(T &x, const T &y) { if (y<x) x=y; }
template<class T> inline void amax(T &x, const T &y) { if (x<y) x=y; }
template<class Iter> void rprintf(const char *fmt, Iter begin, Iter end) {
    for (bool sp=0; begin!=end; ++begin) { if (sp) putchar(' '); else sp = true; printf(fmt, *begin); }
    putchar('\n');
}
class Solution {
public:
    int repeatedNTimes(vector<int>& A) {
    map<int, int> mp;
    EACH (e, A) mp[*e]++;
    EACH (e, mp) if (e->second > 1) return e->first;
    return 0;
    }
};
```


**962. Maximum Width Ramp**

问题描述：Given an array A of integers, a ramp is a tuple (i, j) for which i < j and A[i] <= A[j].  The width of such a ramp is j - i.

Find the maximum width of a ramp in A.  If one doesn't exist, return 0.

Input: [6,0,8,2,1,5]
Output: 4
Explanation: 
The maximum width ramp is achieved at (i, j) = (1, 5): A[1] = 0 and A[5] = 5.

思路：先组成A[i], i的pair，然后按A[i]排序。得到的序列，第i个数的左边都是比A[i]小的，只要找到0-i之间最左边的index，然后用A[i,1]做减法，那么就求得了现在最大的gap。

```cpp
#include<stdio.h>
#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include<string.h>
using namespace std;

typedef long long LL;
typedef vector<int> VI;

#define REP(i,n) for(int i=0, i##_len=(n); i<i##_len; ++i)
#define EACH(i,c) for(__typeof((c).begin()) i=(c).begin(),i##_end=(c).end();i!=i##_end;++i)
#define eprintf(...) fprintf(stderr, __VA_ARGS__)

template<class T> inline void amin(T &x, const T &y) { if (y<x) x=y; }
template<class T> inline void amax(T &x, const T &y) { if (x<y) x=y; }
template<class Iter> void rprintf(const char *fmt, Iter begin, Iter end) {
    for (bool sp=0; begin!=end; ++begin) { if (sp) putchar(' '); else sp = true; printf(fmt, *begin); }
    putchar('\n');
}
class Solution {
public:
    int maxWidthRamp(vector<int>& A) {
    vector<pair<int, int> > t;
    REP (i, A.size()) t.emplace_back(A[i], i);
    sort(t.begin(), t.end());
    int ans = 0;
    int left = t[0].second;
    for (int i=1; i<(int)t.size(); i++) {
        amax(ans, t[i].second - left);
        amin(left, t[i].second);
    }
        
    return ans;
    }
};
```

**963. Minimum Area Rectangle II**

Given a set of points in the xy-plane, determine the minimum area of any rectangle formed from these points, with sides not necessarily parallel to the x and y axes.

If there isn't any rectangle, return 0.

Input: [[1,2],[2,1],[1,0],[0,1]]
Output: 2.00000
Explanation: The minimum area rectangle occurs at [1,2],[2,1],[1,0],[0,1], with an area of 2.

![](https://assets.leetcode.com/uploads/2018/12/21/1a.png)

```cpp
#include<stdio.h>
#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include<string.h>
using namespace std;

typedef long long LL;
typedef vector<int> VI;

#define REP(i,n) for(int i=0, i##_len=(n); i<i##_len; ++i)
#define EACH(i,c) for(__typeof((c).begin()) i=(c).begin(),i##_end=(c).end();i!=i##_end;++i)
#define eprintf(...) fprintf(stderr, __VA_ARGS__)

template<class T> inline void amin(T &x, const T &y) { if (y<x) x=y; }
template<class T> inline void amax(T &x, const T &y) { if (x<y) x=y; }
template<class Iter> void rprintf(const char *fmt, Iter begin, Iter end) {
    for (bool sp=0; begin!=end; ++begin) { if (sp) putchar(' '); else sp = true; printf(fmt, *begin); }
    putchar('\n');
}
class Solution {
public:
    double minAreaFreeRect(vector<vector<int>>& P) {
	int N = P.size();
	bool find = false;
	LL ans = 1LL<<60;
	REP (a, N) REP (b, N) REP (c, N) REP (d, N) {
	    if (a == b || a == c || a == d || b == c || b == d || c == d) continue;
	    if (P[a][0] - P[b][0] != P[d][0] - P[c][0]) continue;
	    if (P[a][1] - P[b][1] != P[d][1] - P[c][1]) continue;

	    LL x1 = P[a][0] - P[b][0];
	    LL x2 = P[c][0] - P[b][0];
	    LL y1 = P[a][1] - P[b][1];
	    LL y2 = P[c][1] - P[b][1];

	    if (x1 * x2 + y1 * y2 == 0) {
		LL area = abs(x1 * y2 - x2 * y1);
		if (area > 0) {
		    find = true;
		    amin(ans, area);
		}
	    }
	}
        
	if (!find) return 0;
	return ans;
    }
};

```

**964. Least Operators to Express Number**

Given a single positive integer x, we will write an expression of the form x (op1) x (op2) x (op3) x ... where each operator op1, op2, etc. is either addition, subtraction, multiplication, or division (+, -, *, or /).  For example, with x = 3, we might write 3 * 3 / 3 + 3 - 3 which is a value of 3.

When writing such an expression, we adhere to the following conventions:

The division operator (/) returns rational numbers.
There are no parentheses placed anywhere.
We use the usual order of operations: multiplication and division happens before addition and subtraction.
It's not allowed to use the unary negation operator (-).  For example, "x - x" is a valid expression as it only uses subtraction, but "-x + x" is not because it uses negation.
We would like to write an expression with the least number of operators such that the expression equals the given target.  Return the least number of expressions used.

Input: x = 3, target = 19

Output: 5

Explanation: 3 * 3 + 3 * 3 + 3 / 3.  The expression contains 5 operations.

Input: x = 5, target = 501

Output: 8

Explanation: 5 * 5 * 5 * 5 - 5 * 5 * 5 + 5 / 5.  The expression contains 8 operations.

Input: x = 100, target = 100000000

Output: 3

Explanation: 100 * 100 * 100 * 100.  The expression contains 3 operations.

Note:

* 2 <= x <= 100
* 1 <= target <= 2 * 10^8

```cpp
#include<unordered_map>
#include<stdio.h>
#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include<string.h>
using namespace std;

typedef long long LL;
typedef vector<int> VI;

#define REP(i,n) for(int i=0, i##_len=(n); i<i##_len; ++i)
#define EACH(i,c) for(__typeof((c).begin()) i=(c).begin(),i##_end=(c).end();i!=i##_end;++i)
#define eprintf(...) fprintf(stderr, __VA_ARGS__)

template<class T> inline void amin(T &x, const T &y) { if (y<x) x=y; }
template<class T> inline void amax(T &x, const T &y) { if (x<y) x=y; }
template<class Iter> void rprintf(const char *fmt, Iter begin, Iter end) {
    for (bool sp=0; begin!=end; ++begin) { if (sp) putchar(' '); else sp = true; printf(fmt, *begin); }
    putchar('\n');
}

class Solution {
public:
    int X;
    unordered_map<int, int> mp;

    int rec(LL target) {
	if (target == 0) return 0;
	auto it = mp.find(target);
	if (it != mp.end()) return it->second;

	LL g = 1;
	int cnt = 0;
	while (g*X < target) {
	    g *= X;
	    cnt++;
	}

	LL guess = rec(target - g) + (cnt == 0? 2: cnt);
	if (g < target && g*X - target < target) {
	    LL tmp = rec(g*X - target) + cnt + 1;
	    amin(guess, tmp);
	}
	mp.emplace(target, guess);
	return guess;
    }

    int leastOpsExpressTarget(int x, int target) {
	mp.clear();
	X = x;
	return rec(target) - 1;
    }
};


```
