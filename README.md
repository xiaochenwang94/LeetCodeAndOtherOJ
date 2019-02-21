# LeetCodeAndOtherOJ

## 背包问题
*0-1背包问题*

leetcode 416 Partition Equal Subset Sum 

问题描述：
给定一个正整数的numlist，划分成加和相等的两部分。

思路：
可以转换为0/1背包问题，加和除以2就是背包的容量。这道题要求背包恰好装满。

```python
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
```

## 其他动态规划问题

leetcode 53 Maximum Subarray

问题描述：给定一个数组，寻找最大的自数组。

思路：e = max(a_i, e)，e代表之前子序列的最大值。

```python
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
```

## 回文串问题
最长回文字串问题。回文串分为两种，奇数长度和偶数长度。例如：aba，abba。为了方便求解在字符串开始、结束、任意两个字符中间添加#号，变成#a#b#a#, #a#b#b#a#。这样做的好处是把奇数和偶数的回文串都转换成了奇数的回文串。暴力算法，根据每个点依次向两边展开，求出最长的回文字串。这种算法的时间复杂度是O(n*n)。字符串变换之后和变换之前的关系如下：(1)原始回文子串的长度等于转换之后的回文字串的半径-1。(2)在变换后的字符串开始添加$符号，变换后的字符串(位置-半径)/2是变换之前回文串开始的位置。

马拉车算法，马拉车算法的思路是基于已经探索的部分，继续探索未探索的部分。例如，在字符串第i个位置，可以利用前面的信息，直接探索半径大于3的范围内是否为回文串。这样便极大的减小了时间复杂度。首先介绍算法使用的符号和变量，id代表当前能延伸到最右端位置的回文串的中心，mx代表当前能延长到最右端的回文串的最右端位置，i代表我们要探索的字符串的位置，j代表i关于id对称的位置。数组p记录了以每个字符为中心的最长回文串的半径。

算法可以分两种情况讨论：(1)i>=mx，此时我们不能从已有的探索中获得任何有用的信息，因此p[i]=1。(2) i < mx，此时又分为两种情况。(2.1)mx-i>p[j]，见下图。i和j关于id对称，而id-mx到id+mx范围内是回文串，也就是p[j]是包含在p[id]内的一个回文字串，所以p[i]在半径为p[j]的范围内也是一个回文串。因此可以在p[j]的基础上进行探索。所以p[i]=p[j]。

![](https://img2018.cnblogs.com/blog/391947/201809/391947-20180916233749134-798948486.png)

(2.2)mx-i<=p[j]，这种情况见下图。以j为中心的回文串的范围超出了以id为中心的回文串，因此我们不知道超出的部分在以i为中心的范围是否还可用。所以p[i]=mx-i。

![](https://img2018.cnblogs.com/blog/391947/201809/391947-20180916233809298-960515229.png)

综上可以归结为下面一行代码
```c++
p[i] = mx>i?min(p[j], mx-i):1
```

leetcode 5 Longest Palindromic Substring

题目描述：给定一个字符串，返回最长的回文字串。

思路：上文的马拉车算法。

```c++
class Solution {
public:
    string longestPalindrome(string s) {
        string ns = "$#";
        for(int i=0;i<s.size();++i){
            ns+=s[i];
            ns+="#";
        }
        vector<int> p(ns.size());
        int max_len = 0, id=0, mx=0, start=0;
        for(int i=1;i<ns.size();++i){
            p[i] = mx>i ? min(p[2*id-i], mx-i) : 1;
            while(ns[i-p[i]] == ns[i+p[i]]) p[i]++;
            if(i+p[i] > mx) {
                mx = i+p[i];
                id = i;
                if(max_len < p[i]) {
                    max_len = p[i];
                    start = (i-p[i])/2;
                }  
            }
        }
        return s.substr(start, max_len-1);
    }
};
```

leetcode 9 Palindrome Number

题目描述：给定一个数字判断是否为回文数字，121是回文数字，但是-121不是。这道题目要求不能转换为字符串。

```c++
class Solution {
public:
    bool isPalindrome(int x) {
        if(x<0) return false;
        int y = 0, xx = x;
        while(xx>0) {
            y=y*10+xx%10;
            xx = xx/10;
        }
        return y==x;
    }
};
```

leetcode 125 Valid Palindrome

题目描述：给定一个字符串，判断是否为回文串，不考虑空格，标点符号。例如"A man, a plan, a canal: Panama"就是一个回文串。

```c++
class Solution {
public:
    bool isPalindrome(string s) {
        string ns="";
        for(int i=0;i<s.size();++i) {
            if((s[i]>='a' && s[i] <='z') || (s[i]>='0' && s[i]<='9')) {
                ns+=s[i];
            } else if (s[i]>='A' && s[i]<='Z') {
                ns+=s[i]-'A'+'a';
            }
        }
        int mid = ns.size()/2;
        for(int i=0;i<=mid;++i){
            if(ns[i] != ns[ns.size()-i-1]) return false;
        }
        return true;
    }
};
```

leetcode 131 Palindrome Partitioning

题目描述：给定一个字符串，求出所有回文字串的划分。例如：输入"aab"，输出为[["aa","b"],["a","a","b"]]。

思路：使用dfs进行递归搜索，每次添加一个字串，如果是回文串继续向后添加，否则不添加。添加进行递归结束之后要清除该串。

```c++
class Solution {
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> result;
        vector<string> v;
        dfs(result, v, s);
        return result;
    }
    
    void dfs(vector<vector<string>> &result, vector<string> v, string s) {
        if(s.size() == 0) {
            result.push_back(v);
            return;
        }
        for(int i=0;i<s.size();++i) {
            if(isPalindrome(s.substr(0, i+1))) {
                v.push_back(s.substr(0, i+1));
                dfs(result, v, s.substr(i+1));
                v.pop_back();
            }
        }
    }
    
    bool isPalindrome(string s) {
        int i=0,j=s.size()-1;
        while(i<=j) {
            if(s[i] != s[j]) return false;
            i++;j--;
        }
        return true;
    }
};
```


## 按顺序刷题

leetcode 4 Median of Two Sorted Arrays

题目描述：给定两个有序数组，找到中位数。时间复杂度O(log(m+n))

思路：参考了别人的解答，先将题目转换为找第K大的数字。找第K大的数字，每次比较两个数组的第K/2的数字。如果第一个大，就丢弃掉第二个数组的前K/2个，并领K=K/2。当K=1的时候，返回两个数组第一个数中小的那个。当某一个数组是空的时候，返回另一个数组的第K个数字即可。

一个tick，中位数等于第n+1/2和第n+2/2个数的平均数。对于基数和偶数同样适用。

```c++
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int k1 = nums1.size();
        int k2 = nums2.size();
        return (findKth(nums1, nums2, 0, 0, (k1+k2+1)/2) + findKth(nums1, nums2, 0, 0, (k1+k2+2)/2)) / 2;
    }
    
    double findKth(vector<int>& nums1, vector<int>& nums2, int i, int j, int k) {
        if(i >= nums1.size()) return nums2[j+k-1];
        if(j >= nums2.size()) return nums1[i+k-1];
        if(k == 1) return min(nums1[i], nums2[j]);
        int m = 0x7fffffff;
        int mid1 = i+k/2-1 >= nums1.size() ? m : nums1[i+k/2-1];
        int mid2 = j+k/2-1 >= nums2.size() ? m : nums2[j+k/2-1];
        if(mid1 > mid2) {
            return findKth(nums1, nums2, i, j+k/2, k-k/2);
        } else {
            return findKth(nums1, nums2, i+k/2, j, k-k/2);
        }
    }
};
```

leetcode 10 Regular Expression Matching

题目描述：实现正则表达式的*和.

思路：完全参照了参考答案

- 若p为空，若s也为空，返回true，反之返回false。

- 若p的长度为1，若s长度也为1，且相同或是p为'.'则返回true，反之返回false。

- 若p的第二个字符不为*，若此时s为空返回false，否则判断首字符是否匹配，且从各自的第二个字符开始调用递归函数匹配。

- 若p的第二个字符为*，进行下列循环，条件是若s不为空且首字符匹配（包括p[0]为点），调用递归函数匹配s和去掉前两个字符的p（这样做的原因是假设此时的星号的作用是让前面的字符出现0次，验证是否匹配），若匹配返回true，否则s去掉首字母（因为此时首字母匹配了，我们可以去掉s的首字母，而p由于星号的作用，可以有任意个首字母，所以不需要去掉），继续进行循环。

- 返回调用递归函数匹配s和去掉前两个字符的p的结果（这么做的原因是处理星号无法匹配的内容，比如s="ab", p="a*b"，直接进入while循环后，我们发现"ab"和"b"不匹配，所以s变成"b"，那么此时跳出循环后，就到最后的return来比较"b"和"b"了，返回true。再举个例子，比如s="", p="a*"，由于s为空，不会进入任何的if和while，只能到最后的return来比较了，返回true，正确）。

```c++
class Solution {
public:
    bool isMatch(string s, string p) {
        if (p.empty()) return s.empty();
        if (p.size() == 1) {
            return (s.size() == 1 && (s[0] == p[0] || p[0] == '.'));
        }
        if (p[1] != '*') {
            if (s.empty()) return false;
            return (s[0] == p[0] || p[0] == '.') && isMatch(s.substr(1), p.substr(1));
        }
        while (!s.empty() && (s[0] == p[0] || p[0] == '.')) {
            if (isMatch(s, p.substr(2))) return true;
            s = s.substr(1);
        }
        return isMatch(s, p.substr(2));
    }
};
```

leetcode 11 Container With Most Water    

问题描述：找出能装的最多水的两个柱子。

思路：开始想复杂了，设置两个指针，一个自左向右，另一个自右向左。如果height[i] < height[j] i++ else j--; 一个小优化，当移动一个柱子的时候，可以连续移动到一个比当前大的柱子。

```python
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
```

leetcode 12 Integer to Roman  

问题描述：整数转换为罗马数字

思路：和进制转换一样，把特殊的情况（4, 9, 40, 90, 400, 900)考虑进去就行了。

```python
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
```

leetcode 13 Roman to Integer   

问题描述：罗马数字转换为整数

思路: 和上一题相反，一个规律就是特殊情况不会出现在正常情况前面，例如：V总是出现在IV前，出现了IV之后就不可能出现V。顺序扫描相加即可。

```python
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
```
           
leetcode 14 Longest Common Prefix

问题描述：给定一个字符串的list，找出最长的前缀。例如：
    
    Input: ["flower","flow","flight"]
    Output: "fl"
思路：直接便利寻找即可。这道题还有一种巧妙的解法，给字符串排序，之后只需要比较第一个和最后一个字符串即可。

```python
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
```

leetcode 15 3Sum

问题描述：找出数组中三个相加为0的数字，返回所有结果。

思路：这道题参考了答案，弄个半个小时才通过。开始使用暴力求解，果断超时。解法中先把列表排序，然后从左到右扫一次，每次固定一个数字，找另外两个数字的和与当前数字相反。当扫到正数就可以结束了，因子正数后面都是正数，不可能找到结果。

在找后面两个数字的时候，左边的数字时i=idx+1，右边的是j=size-1。当i < j的时候寻找。如果找到了两个数字，那么添加到结果中，之后进行剪枝，当nums[i] == nums[i+1], nums[j] == nums[j-1]，就过滤掉，不这样求出的结果会重复。

找完后面两个数字之后，同样的固定的数字也进行剪枝。

``` python
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
```

leetcode 16 3Sum Closest

问题描述: 和上一道题相似，现在要求加和和给定的target最接近。

思路：和上一题一样，固定一个数字，找另外两个数字。使用两个index从左到右，从右到左，比三层循环次数要少，可以加速。其他剪枝方法不太好用。

``` python
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
```

leetcode 18 4Sum

问题描述：四个数字加和

思路：和前一道题目一样，套一层for循环就行了。

``` python
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
```                    

                    
leetcode 19 Remove Nth Node From End of List

问题描述：删除倒数第n个数字

思路：两个指针，第一个先跑n个，第二个再跑。然后删除。

``` python
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
```

leetcode 21 Merge Two Sorted Lists

问题描述：合并两个有序的链表

思路：参考了大神的代码，递归写法。每次比较l1, l2如果l1.val > l2.val 那么交换两段列表，然后l1向后走一个。

``` python
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
```

leetcode 22 Generate Parentheses

问题描述：给定n对括号，求出所有可能。

思路：

解法一：BFS进行搜索，每次分两种可能，添加左括号或者右括号。

``` python
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
```

解法2: dfs

``` python
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
```

leetcode 29 Divide Two Integers

问题描述：给定两个正数，求整除之后的结果。不能用乘法，除法，mod。

思路：使用递归模拟，当除数大于被除数直接返回，否则加上初始被除数的倍数。**这道题应该注意题目描述的边界条件，以及两个数符号相反的情况！**

``` python
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
```

leetcode 48 Rotate Image

题目描述：给定一个矩阵，顺时针旋转90度。要inplace。

思路：可以按照规律推出每个点的位置，有些复杂。简单的做法是，先做转置，之后再沿着y轴翻转。延伸一下，如果是逆时针旋转90度，转置之后沿着x轴翻转即可。旋转180度，转置之后沿着x=y翻转。

```c++
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n=matrix.size();
        for(int i=0;i<n;++i) {
            for(int j=i;j<n;++j) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }    
        for(int i=0;i<n;++i) {
            for(int j=0;j<n/2;++j){
                swap(matrix[i][j], matrix[i][n-j-1]);
            }
        }
    }
};
```

leetcode 54 Spiral Matrix

问题描述：旋转输出矩阵

思路：使用四个变量top, right, down, left控制输出。

```cpp
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> ret;
        if(matrix.size() == 0) return ret;
        int left=0, right=matrix[0].size()-1, top=0, down=matrix.size()-1;
        int i=0, j=0;
        while(true) {
            if(right < left) break;
            while(j<=right) ret.push_back(matrix[i][j++]);
            j--; top++; i++;
            if(down < top) break;
            while(i<=down) ret.push_back(matrix[i++][j]);
            i--; right--; j--;
            if(right < left) break;
            while(j>=left) ret.push_back(matrix[i][j--]);
            j++; down--; i--;
            if(down < top) break;
            while(i>=top) ret.push_back(matrix[i--][j]);
            i++; left++; j++;
            
        }
        return ret;
    }
};
```

leetcode 58 Merge Intervals

题目描述：合并区间

思路：排序之后合并

```cpp
/**
 * Definition for an interval.
 * struct Interval {
 *     int start;
 *     int end;
 *     Interval() : start(0), end(0) {}
 *     Interval(int s, int e) : start(s), end(e) {}
 * };
 */

bool cmp(const Interval& x, const Interval& y) {
        return x.start < y.start;
}

class Solution {
public:
    vector<Interval> merge(vector<Interval>& intervals) {
        vector<Interval> ret;
        if(intervals.size() == 0) return ret;
        sort(intervals.begin(), intervals.end(), cmp);
        int end=intervals[0].end;
        int start=intervals[0].start;
        ret.push_back(intervals[0]);
        for(int i=1;i<intervals.size();++i) {
            Interval tmp = ret.back();
            ret.pop_back();
            if(intervals[i].start <= tmp.end) {
                Interval new_tmp(tmp.start, max(tmp.end, intervals[i].end));
                ret.push_back(new_tmp);
            } else {
                ret.push_back(tmp);
                ret.push_back(intervals[i]);
            }
        }
        return ret;
    }
};
```

leetcode 60 Permutation Sequence

题目描述：给定n和k，数字由[1,2,3,...,n]组成，求第k小的数。

思路：跳跃查找。

```cpp
class Solution {
public:
    string getPermutation(int n, int k) {
        string ret="";
        if(n == 0) return ret;
        vector<int> v;
        int prod = 1;
        for(int i=1;i<=n;++i) {
            ret += i + '0';
            prod *= i;
            v.push_back(i);
        }
        int i=0;
        k--;
        while(i<n) {
            prod /= n-i;
            ret[i] = v[k / prod]+'0';
            v.erase(k/prod+v.begin());
            i++;
            k = k % prod;
        }
        return ret;
    }
};
```

leetcode 61 Rotate List

题目描述：从第k个元素断开，重新拼接链表

思路：如果链表长度为3，旋转3次相当于没有旋转。所以使用k%len，len为链表的长度。然后使用快慢指针确定断开的地方。

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        ListNode *p=head, *q=head;
        int len=0;
        while(p) {
            len++;
            p=p->next;
        }
        if(len == 0) {
            return head;
        }
        int step = k%len;
        p=head;
        while(step>0) {
            p=p->next;
            step--;
        }
        while(p && p->next) {
            q=q->next;
            p=p->next;
        }
        if(p) {
            p->next = head;
            head = q->next;
            q->next=NULL;
        }
        return head;
    }
};
```

leetcode 62 Unique Paths

题目描述：机器人只能向下或者向右走，求从左上到右下的方法数量。

思路：二维动归，只需要左边和上边，一维也可以。

```cpp
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> v(m, vector<int>(n, 0));
        cout<<v.size()<<endl;
        cout<<v[0].size()<<endl;
        for(int i=0;i<m;++i) {
            v[i][0] = 1;
        }
        for(int i=0;i<n;++i) {
            v[0][i] = 1;
        }
        for(int i=1;i<m;++i) {
            for(int j=1;j<n;++j) {
                v[i][j] = v[i-1][j] + v[i][j-1];
            }
        }
        return v[m-1][n-1];
    }
};

class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<int> v(n, 0);
        for(int i=0;i<n;++i) {
            v[i] = 1;
        }
        for(int i=1;i<m;++i) {
            for(int j=1;j<n;++j) {
                v[j] += v[j-1];
            }
        }
        return v[n-1];
    }
};
```

leetcode 63 Unique Paths II

题目描述：和上一题类似，但是路线上有障碍。

思路：如果当前格子有障碍，那么通行的方式变成0即可。

```cpp
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        vector<long long int> v(obstacleGrid[0].size(), 0);
        for(int i=0;i<obstacleGrid[0].size();++i) {
            if(obstacleGrid[0][i] != 1) {
                v[i] = 1;
            } else {
                break;
            }
        }
        for(int i=1;i<obstacleGrid.size();++i) {
            for(int j=0;j<obstacleGrid[0].size();++j) {
                if(obstacleGrid[i][j] == 1) {
                    v[j] = 0;
                } else {
                    if(j != 0)
                        v[j] += v[j-1];    
                    else
                        v[j] = obstacleGrid[i][j] == 1 ? 0 : v[j];
                }
            }
        }
        return v[obstacleGrid[0].size()-1];
    }
};
```

leetcode 64 Minimum Path Sum

题目描述：给定一个路径矩阵，求从左上到右下的最短距离，只能向右或者向下走。

思路：动归。

```cpp
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int path = 0;
        int m=grid.size(), n=grid[0].size();
        vector<long long int> v(n, 0);
        for(int i=0;i<n;++i) {
            if(i>0) v[i] = grid[0][i] + v[i-1];
            else v[i] = grid[0][i];
        }
        for(int i=1;i<m;++i) {
            for(int j=0;j<n;++j) {
                if(j == 0) {
                    v[j] = grid[i][0] + v[j];
                    continue;
                }
                v[j] = min(v[j], v[j-1]) + grid[i][j];
            }
        }
        return v[n-1];
    }
};
```

leetcode 98 Validate Binary Search Tree

问题描述：给定一棵二叉搜索树，判断是否合法

思路：按照二叉搜索树的性质判断

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        return !root || dfs(root, LONG_MIN, LONG_MAX);
    }
    
    bool dfs(TreeNode* root, long low, long high) {
        if(!root) {
            return true;
        } 
        if(root->val <= low || root->val >= high) {
            return false;
        }
        return dfs(root->left, low, root->val) && dfs(root->right, root->val, high);
    }
};
```

leetcode 67 Add Binary

问题描述：二进制大整数相加

思路：按顺序相加即可，写法值得记录。

```cpp
class Solution {
public:
    string addBinary(string a, string b) {
        int len1 = a.size();
        int len2 = b.size();
        string sum = "";
        int carry = 0;
        while(len1 > 0 || len2 > 0) {
            int first = len1 <= 0 ? 0 : a[len1-1]-'0';
            int second = len2 <= 0 ? 0 : b[len2-1]-'0';
            int tsum = first + second + carry;
            carry = tsum / 2;
            sum = (char)(tsum%2+'0') + sum;
            len1--; len2--;
        }
        if(carry > 0) sum = '1' + sum;
        return sum;
    }
};
```

leetcode 69 Sqrt(x)

问题描述：给定x，返回sqrt(x)的整数部分。

思路：

1. 从小到大遍历小于x的数字，知道i*i>x，返回i-1;

```cpp
class Solution {
public:
    int mySqrt(int x) {
        for(long long int i=1;i<=x;++i) {
            if(i*i>x) return i-1;
            if(i*i == x) return i;
        }
        return x;
    }
};
```

2. 

leetcode 73 Set Matrix Zeroes

问题描述：矩阵中如果某个数字为0，把它所在的行和列都变成0，要求inplace变化。

思路：使用一个行长度和列长度的vector，如果某个数字为0，把row，col位置变成0。最后统一变化。

```cpp
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        vector<int> row(matrix.size(), 1), col(matrix[0].size(), 1);
        for(int i=0;i<matrix.size();++i) {
            for(int j=0;j<matrix[i].size();++j) {
                if(matrix[i][j] == 0) {
                    row[i] = 0;
                    col[j] = 0;
                }
            }
        }
        for(int i=0;i<row.size();++i) {
            if(row[i] == 0) {
                for(int j=0;j<matrix[i].size();++j) {
                    matrix[i][j] = 0;
                }
            }
        }
        for(int i=0;i<col.size();++i) {
            if(col[i] == 0) {
                for(int j=0;j<matrix.size();++j) {
                    matrix[j][i] = 0;
                }
            }
        }
    }
};
```

leetcode 74 Search a 2D Matrix

问题描述：给定一个矩阵，矩阵的每一行递增，每一列递增，前一行的最大数字小于本行的最小数字。查找给定的数字是否在矩阵中。

思路：从左下角查询，如果x比matrix[i][j]大，向右，小向左。和240题目完全相同。

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int i=matrix.size()-1, j=0;
        while(i>=0 && j<matrix[0].size()) {
            if(target == matrix[i][j]) return true;
            if(target > matrix[i][j]) {
                j++;
            } else {
                i--;
            }
        }
        return false;
    }
};
```

leetcode 75 Sort Colors

问题描述：给定一个0，1，2的序列，对其进行排序。

思路：
1. 使用Counting Sort，把每个数字数一遍，然后写一遍。

```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        vector<int> cnts(3, 0);
        int size = nums.size();
        for(int i=0;i<size;++i) {
            cnts[nums[i]]++;
        }
        for(int i=0;i<size;++i) {
            if(i<cnts[0]) {
                nums[i] = 0;
            } else if(i < cnts[0] + cnts[1]) {
                nums[i] = 1;
            } else {
                nums[i] = 2;
            }
        }
    }
};
```
2. 可以只扫描一遍，使用0，2指针分别指向0的位置和2的位置。如果当前扫描到0和0指针交换，扫描到2和2指针交换。

```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int size=nums.size(), zero=0, two=size-1;
        for(int i=0;i<=two;++i) {
            if(nums[i] == 0) {
                swap(nums[i], nums[zero++]);
            } else if (nums[i] == 2) {
                swap(nums[i--], nums[two--]);
            }
        }
    }
};
```

leetcode 78 Subsets

题目描述：给定一个集合，求出集合的所有子集。

思路：
1. 针对每一个数字有两种策略，取或者不取。使用dfs进行递归。

```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> cur;
        dfs(res, cur, 0, nums);
        return res;
    }
    
    void dfs(vector<vector<int>> &res, vector<int> cur, int idx, vector<int> &nums) {
        if(idx == nums.size()) {
            res.push_back(cur);
            return ;
        } 
        cur.push_back(nums[idx]);
        dfs(res, cur, idx+1, nums);
        cur.pop_back();
        dfs(res, cur, idx+1, nums);
    }
};
```

2. 我们可以一位一位的网上叠加，比如对于题目中给的例子[1,2,3]来说，最开始是空集，那么我们现在要处理1，就在空集上加1，为[1]，现在我们有两个自己[]和[1]，下面我们来处理2，我们在之前的子集基础上，每个都加个2，可以分别得到[2]，[1, 2]，那么现在所有的子集合为[], [1], [2], [1, 2]，同理处理3的情况可得[3], [1, 3], [2, 3], [1, 2, 3], 再加上之前的子集就是所有的子集合

```cpp
class Solution {
public:
    vector<vector<int> > subsets(vector<int> &S) {
        vector<vector<int> > res(1);
        sort(S.begin(), S.end());
        for (int i = 0; i < S.size(); ++i) {
            int size = res.size();
            for (int j = 0; j < size; ++j) {
                res.push_back(res[j]);
                res.back().push_back(S[i]);
            }
        }
        return res;
    }
};
```

3. 把数组中所有的数分配一个状态，true表示这个数在子集中出现，false表示在子集中不出现，那么对于一个长度为n的数组，每个数字都有出现与不出现两种情况，所以共有2n中情况，那么我们把每种情况都转换出来就是子集了，我们还是用题目中的例子, [1 2 3]这个数组共有8个子集，每个子集的序号的二进制表示，把是1的位对应原数组中的数字取出来就是一个子集，八种情况都取出来就是所有的子集了

```cpp
class Solution {
public:
    vector<vector<int> > subsets(vector<int> &S) {
        vector<vector<int> > res;
        sort(S.begin(), S.end());
        int max = 1 << S.size();
        for (int k = 0; k < max; ++k) {
            vector<int> out = convertIntToSet(S, k);
            res.push_back(out);
        }
        return res;
    }
    vector<int> convertIntToSet(vector<int> &S, int k) {
        vector<int> sub;
        int idx = 0;
        for (int i = k; i > 0; i >>= 1) {
            if ((i & 1) == 1) {
                sub.push_back(S[idx]);
            }
            ++idx;
        }
        return sub;
    }
};
```

leetcode 79 Word Search

问题描述：给定一个字符矩阵和一个单词，查找单词是否在矩阵中。

思路：图的dfs搜索。

```cpp
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        int m = board.size(), n = board[0].size();
        vector<vector<bool>> isVisited(m, vector<bool>(n, false));
        for(int i=0;i<m;++i) {
            for(int j=0;j<n;++j) {
                if(board[i][j] == word[0]) {
                    isVisited[i][j] = true;
                    if(dfs(board, word.substr(1), isVisited, i, j)) {
                        return true;
                    } 
                    isVisited[i][j] = false;
                }
            }
        }
        return false;
    }
    
    bool dfs(vector<vector<char>>& board, string word, vector<vector<bool>> &isVisited, int i, int j) {
        if(word.size() == 0) {
            return true;
        }
        bool res = false;
        if(i>0 && !isVisited[i-1][j] && word[0] == board[i-1][j]) {
            isVisited[i-1][j] = true;
            res = dfs(board, word.substr(1), isVisited, i-1, j);
            isVisited[i-1][j] = false;
        }
        if(res) {
                return true;
        }
        if(j>0 && !isVisited[i][j-1] && word[0] == board[i][j-1]) {
            isVisited[i][j-1] = true;
            res = dfs(board, word.substr(1), isVisited, i, j-1);
            isVisited[i][j-1] = false;
        }
        if(res) {
            return true;
        }
        if(i<board.size()-1 && !isVisited[i+1][j] && word[0] == board[i+1][j]) {
            isVisited[i+1][j] = true;
            res = dfs(board, word.substr(1), isVisited, i+1, j);
            isVisited[i+1][j] = false;
        }
        if(res) {
            return true;
        }
        if(j<board[0].size()-1 && !isVisited[i][j+1] && word[0] == board[i][j+1]) {
            isVisited[i][j+1] = true;
            res = dfs(board, word.substr(1), isVisited, i, j+1);
            isVisited[i][j+1] = false;
        }
        return res;
    }
};
```

leetcode 80 Remove Duplicates from Sorted Array II

问题描述：给定一个有序的数组，去除出现一次以上的数字。

思路：使用快慢指针，慢指针指向保留的位置，如果重复超过一次，那么不保留，快指针指向下一个。

```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int res = 1, cnt=1, size=nums.size();
        if(size == 0) return 0;
        int old = nums[0]; 
        for(int i=1;i<size;++i) {
            if(nums[i] == old && cnt == 1) {
                nums[res++] = nums[i];
                cnt--;
            } else if(nums[i] != old) {
                nums[res++] = nums[i];
                cnt = 1;
            }
            old=nums[i];
        }
        return res;
    }
};
```

leetcode 94 Binary Tree Inorder Traversal

问题描述：中序遍历

思路：递归

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        dfs(res, root);
        return res;
    }
    
    void dfs(vector<int> &res, TreeNode* root) {
        if(!root) {
            return;
        }
        dfs(res, root->left);
        res.push_back(root->val);
        dfs(res, root->right);
    }
};
```

leetcode 96 Unique Binary Search Trees

问题描述：给定n，求1~n的数字能够组成的BST的种数。

思路：动归，以k为根，那么左子树是1~k-1，右子树是k+1 ~n。那么num[k]=num[k-1]+num[n-k-1+1]。归纳公式如下：$$C_0=1 \ \  and \ \ C_{n+1}=\sum_{i=0}^nC_iC_{n-i}$$

以上公式就是卡塔兰数。

```cpp
class Solution {
public:
    int numTrees(int n) {
        vector<int> v;
        v.push_back(1);
        v.push_back(1);
        for(int i=2;i<=n;++i) {
            int num=0;
            for(int j=0;j<i;++j) {
                num+=v[j]*v[i-1-j];
            }
            v.push_back(num);
        }
        return v[n];
    }
    
};
```

leetcode 100 Same Tree

问题描述：给定两棵树，判断是不是相等的两棵树。

思路：开始想按序遍历，后来发现需要在遍历的时候考虑null的问题。要不然不同的两棵树遍历的结果可能相同。例如：[1,1],[1,null,1]。可以同时遍历两棵树，只要结果不同就可以返回False了。

``` python
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
```

leetcode 388 Longest Absolute File Path

问题描述：给定一个表示路径的字符串，找到其中最长的文件路径。

思路：开始看到文件路径的题目就想到了递归，但是想了很久没有想出来，参考了别人的代码。使用一个map记录每一个level的长度。例如最开始在第一层，那么就使用level=0的长度，代表到达这一层之前的长度。如果在这一层找到了文件，也就是包含.。那么就更新最长的长度。如果没有，那么就是文件夹，把当前的长度给level+1。比较简单的处理方法使用了istringstream可以简单处理掉\n。另外\t或者\n是一个字符长度。

```c++
class Solution {
public:
    int lengthLongestPath(string input) {
        int res = 0;
        istringstream iss(input);
        unordered_map<int, int> m;
        string line="";
        while(getline(iss, line)) {
            int level = line.find_last_of('\t') + 1;
            int size = line.substr(level).size();
            if(line.find('.') != string::npos){
                res = max(res, m[level]+size);
            } else {
                m[level+1] = size + m[level]+1;
            }
        }
        return res;
    }
};
```

leetcode 415. Add Strings

题目描述：字符串加法

思路：按位加，学习到了简洁的写法

```c++
class Solution {
public:
    string addStrings(string num1, string num2) {
        int i=num1.size()-1, j=num2.size()-1;
        int carry = 0;
        string result = "";
        while(i>=0||j>=0) {
            int a = i>=0 ? num1[i--] - '0' : 0;
            int b = j>=0 ? num2[j--] - '0' : 0;
            int r = a + b + carry;
            carry = r / 10;
            result = (char)(r % 10 + '0') + result;
        }
        return carry>0?(char)(carry+'0')+result:result;
        
    }
};
```

leetcode 796. Rotate String

题目描述：给定两个字符串，判断A能否通过若干次移动变成B

思路：令C=A+A，如果C中能够找到B那么A就可以通过若干次移动变成C。

```c++
class Solution {
public:
    bool rotateString(string A, string B) {
        string c = A+A;
        return A.size() == B.size() && c.find(B) != c.npos;
    }
};
```

leetcode 812. Largest Triangle Area

题目描述：给定一系列点，求面积最大的三角形

思路：套公式如下：
![](https://images2018.cnblogs.com/blog/391947/201808/391947-20180827003457034-465504228.jpg)

```c++
class Solution {
public:
    double largestTriangleArea(vector<vector<int>>& points) {
        double max_len = 0;
        for(int i=0;i<points.size()-2;++i) {
            for(int j=i+1;j<points.size()-1;++j) {
                for(int k=j+1;k<points.size();++k) {
                    int p1x = points[i][0], p1y = points[i][1];
                    int p2x = points[j][0], p2y = points[j][1];
                    int p3x = points[k][0], p3y = points[k][1];
                    double size = p1x*p2y-p2x*p1y+p2x*p3y-p2y*p3x+p3x*p1y-p3y*p1x;
                    size = abs(size) / 2;
                    if(size>max_len) {
                        max_len = size;
                    }
                }
            }
        }
        return max_len;
    }
};
```


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

思路：[参考连接](https://blog.csdn.net/fuxuemingzhu/article/details/85223775)

长方形有两个性质，满足就是长方形
* 长方形的两条对角线长度相等
* 长方形的两条对角线互相平分（中点重合）

具体做法是，我们求出任意两个点构成的线段的长度（的平方）、线段的中心坐标，然后用字典保存成（长度，中心点x，中心点y）：[(线段1起点，线段1终点)， (线段2起点，线段2终点)……]。把这个字典弄好了之后，我们需要对字典做一次遍历，依次遍历相同长度和中心点的两个线段构成的长方形的面积，保留最小值就好了。

知道两条对角线坐标，求长方形的面积，方法是找两条临边，然后相乘即可。

```python
class Solution(object):
    def minAreaFreeRect(self, points):
        """
        :type points: List[List[int]]
        :rtype: float
        """
        N = len(points)
        # (l^2, x#, y#) : [(0,1), (1,2)]
        d = collections.defaultdict(list)
        for i in range(N - 1):
            pi = points[i]
            for j in range(i + 1, N):
                pj = points[j]
                l = (pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2
                x = (pi[0] + pj[0]) / 2.0
                y = (pi[1] + pj[1]) / 2.0
                d[(l, x, y)].append((i, j))
        res = float("inf")
        for l in d.values():
            M = len(l)
            for i in range(M - 1):
                p0, p2 = points[l[i][0]], points[l[i][1]]
                for j in range(i + 1, M):
                    p1, p3 = points[l[j][0]], points[l[j][1]]
                    d1 = math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)
                    d2 = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                    area = d1 * d2
                    res = min(res, area)
        return 0 if res == float('inf') else res
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
