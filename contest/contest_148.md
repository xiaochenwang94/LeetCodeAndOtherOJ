# Leetcode Contest 148 题解
## [第一题  Decrease Elements To Make Array Zigzag](https://leetcode.com/contest/weekly-contest-148/problems/decrease-elements-to-make-array-zigzag/)
题目描述：给定一个数组，可以不断对数组中的元素做**减小**操作，每次减小1。

一个数组是Zigzag数组的定义如下：
* A[0] > A[1] < A[2] > A[3] < A[4] > ...
* A[0] < A[1] > A[2] < A[3] > A[4] < ...

求：最少进行多少次减小操作能够让数组变成Zigzag数组。

```
Input: nums = [1,2,3]
Output: 2
Explanation: We can decrease 2 to 0 or 3 to 1.

Input: nums = [9,6,1,6,2]
Output: 4
```

思路：Zigzag数组一共有两种表现形式，1)大小大；2)小大小。尝试两种方式，看哪一种方式消耗的次数少。

无论哪一种方式，可以归结为，改变奇数位，让奇数位小于偶数位；改变偶数位，让偶数位小于奇数位。

参考LeetCode neal_wu的代码
```cpp
const int INF = 1e9;
class Solution {
public:
    int movesToMakeZigzag(vector<int>& nums) {
        int sum[2] = {0}, n = nums.size();
        for(int i=0; i<n; ++i) {
            int left = i == 0 ? INF : nums[i-1];
            int right = i == n-1 ? INF : nums[i+1];
            int goal = min(left, right) - 1;
            if(nums[i] > goal)
                sum[i % 2] += nums[i] - goal;
        }
        return min(sum[0], sum[1]);
    }
};
```

## [第二题 Binary Tree Coloring Game](https://leetcode.com/contest/weekly-contest-148/problems/binary-tree-coloring-game/)
题目描述：给定一棵二叉树，节点的值从1-n，且n为奇数。两个人分别做涂色操作。玩家一红色，玩家二蓝色。当玩家一选定开始涂色的位置后，玩家二也选一个位置开始涂色。涂色可以给当前节点的子节点和父节点涂色。问玩家二能否选择一个点，最终涂色的节点比玩家一多？

![](https://assets.leetcode.com/uploads/2019/08/01/1480-binary-tree-coloring-game.png)
```
Input: root = [1,2,3,4,5,6,7,8,9,10,11], n = 11, x = 3
Output: true
Explanation: The second player can choose the node with value 2.
```

思路：这道题比赛的时候完全没有理解题目的意思，现在看来是只要玩家一选定一个点之后，玩家二选定另一个点，把玩家一的路堵住，使得玩家二能遍历到的点超过一半。那么就要求出每个点的子节点的数量，放到一个map m中。假定x是玩家一选定的开始节点，那么有三种情况玩家二可以获胜。1) m[x] <= n/2，那么玩家二只要选择x的父节点即可。因为节点的数量是奇数个那么玩家二能遍历到的节点的数量一定多于n/2；2&3) m[x->left] > n/2 || m[x->right] > n/2。这时玩家二选择大的子节点一边就可以。

参考Leetcode neal_wu的代码
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
    unordered_map<int, TreeNode*> f;
    unordered_map<TreeNode*, int> m;
    
    void dfs(TreeNode* root) {
        if(!root) return;
        f[root->val] = root;
        dfs(root->left);
        dfs(root->right);
        m[root] = m[root->left] + m[root->right] + 1;
    }
    
    bool btreeGameWinningMove(TreeNode* root, int n, int x) {
        dfs(root);
        return m[f[x]] <= n/2 || m[f[x]->left] > n/2 || m[f[x]->right] > n/2;
    }
};
```

## [第三题 Snapshot Array](https://leetcode.com/contest/weekly-contest-148/problems/snapshot-array/) 
题目描述：实现一个快照数组，包含以下功能：

* `SnapshotArray(int length)`初始化一个数组，长度为length。每一个元素都是0.
* `void set(index, val)`把数组的第index位设置为val
* `int snap()`制作一个快照，返回快照的id，id是调用这个函数的次数减1
* `int get(index, snap_id)`获取快照snap_id中index位置的val

```
Input: ["SnapshotArray","set","snap","set","get"]
[[3],[0,5],[],[0,6],[0,0]]
Output: [null,null,0,null,5]
Explanation: 
SnapshotArray snapshotArr = new SnapshotArray(3); // set the length to be 3
snapshotArr.set(0,5);  // Set array[0] = 5
snapshotArr.snap();  // Take a snapshot, return snap_id = 0
snapshotArr.set(0,6);
snapshotArr.get(0,0);  // Get the value of array[0] with snap_id = 0, return 5
```

思路：使用一个map，每次记录每个snapshot的改变，查询的时候依次从snap_id向前查，直到找到上一次index出现的snapshot，返回这个值。如果一直没有查到，就是没有改变过，那么返回0。

```cpp
class SnapshotArray {
public:
    vector<int> nums;
    vector<unordered_map<int, int>> vm;
    unordered_map<int, int> m;
    int id;
    SnapshotArray(int length) {
        nums.resize(length);
        id = 0;
    }
    
    void set(int index, int val) {
        m[index] = val;
    }
    
    int snap() {
        vm.push_back(m);
        m.clear();
        id++;
        return id-1;
    }
    
    int get(int index, int snap_id) {
        while(snap_id >= 0 && vm[snap_id].find(index) == vm[snap_id].end()) {
            snap_id--;
        }
        if(snap_id < 0) return 0;
        return vm[snap_id][index];
    }
};

/**
 * Your SnapshotArray object will be instantiated and called as such:
 * SnapshotArray* obj = new SnapshotArray(length);
 * obj->set(index,val);
 * int param_2 = obj->snap();
 * int param_3 = obj->get(index,snap_id);
 */
```

这种解法有一个缺点，如果一直没有改变某一个index，然后做snapshot，那么查询路径就会很长。查询的时间复杂度就是O(n)。参考Leetcode neal_wu的代码，有一种O(log(n))的解法。使用一个二维的vector v，大小是(length, ?)，每次有改变就把改变的值放到相应的位置。例如：set(1, 10)，那么就在v[1]的后面添加{snap_id, 10}。在查找的时候，在对应index位置使用二分查找，找到第一个小于等于snap_id的值即可。对于pair，如果第一个值相同那么就比较第二个值。

```cpp
const int INF = 1e9 + 7;
class SnapshotArray {
public:
    vector<vector<pair<int, int>>> v;
    int n, time;
    SnapshotArray(int length) {
        n = length;
        time = 0;
        v.assign(n, vector<pair<int, int>>(1, {0, 0}));
    }
    
    void set(int index, int val) {
        v[index].push_back({time, val});
    }
    
    int snap() {
        return time++;
    }
    
    int get(int index, int snap_id) {
        int p = lower_bound(v[index].begin(), v[index].end(), make_pair(snap_id, INF)) - v[index].begin();
        p--;
        return v[index][p].second;
    }
};

/**
 * Your SnapshotArray object will be instantiated and called as such:
 * SnapshotArray* obj = new SnapshotArray(length);
 * obj->set(index,val);
 * int param_2 = obj->snap();
 * int param_3 = obj->get(index,snap_id);
 */
```

## [第四题 Longest Chunked Palindrome Decomposition](https://leetcode.com/contest/weekly-contest-148/problems/longest-chunked-palindrome-decomposition/)
题目描述：给定一个字符串，把字符串分成k份，a_1, a_2, ..., a_k，满足对于
1 <=i <= k a_i = a_{k-i}

例如：a_0 = a_k, a_1 = a_k-1。

返回最大的k

```
Input: text = "ghiabcdefhelloadamhelloabcdefghi"
Output: 7
Explanation: We can split the string on "(ghi)(abcdef)(hello)(adam)(hello)(abcdef)(ghi)".

Input: text = "merchant"
Output: 1
Explanation: We can split the string on "(merchant)".

Input: text = "antaprezatepzapreanta"
Output: 11
Explanation: We can split the string on "(a)(nt)(a)(pre)(za)(tpe)(za)(pre)(a)(nt)(a)".

Input: text = "aaa"
Output: 3
Explanation: We can split the string on "(a)(a)(a)".
```

思路：对于位置pos，对称的位置是n-pos。字串的长度为len，那么我们要判断[pos, pos+len]和[n-pos, n-pos+len]是否对称。使用一个dp数组记录已经探索过的部分，dp[i]表示，从i位置开始，到n-i的字符串k最大是多少。

参考Leetcode neal_wu的代码
```cpp
class Solution {
public:
    vector<int> dp;
    
    int dfs(string text, int start) {
        int end = text.size() - start;
        if(start >= end) return 0;
        if(dp[start] != -1) return dp[start];
        int &ans = dp[start];
        ans = 1;
        for(int len=1; start+len <= end-len; ++len) {
            if(text.compare(start, len, text, end-len, len) == 0) {
                ans = max(ans, dfs(text, start+len) + 2);
            }
        }
        return ans;
    }
    
    int longestDecomposition(string text) {
        dp.clear();
        dp.assign(text.size()+1, -1);
        return dfs(text, 0);
    }
};
```