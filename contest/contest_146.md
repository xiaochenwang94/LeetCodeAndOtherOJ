# Leetcode Contest 146 题解
## [第一题 Number of Equivalent Domino Pairs](https://leetcode.com/contest/weekly-contest-146/problems/number-of-equivalent-domino-pairs/)

题目描述：有一列多米诺骨牌，每个牌有两个数字，dominoes[i]=[a, b]和dominoes[j]=[c, d]相等，如果a==c&&b==d或者a==d&&b==c。求出有多少对相等的，且1 <= i < j <= dominoes.length

思路：任意两个相等必定是一大一小，找出有多少个相同的，然后组合一下就可以了。

```cpp
class Solution {
public:
    int numEquivDominoPairs(vector<vector<int>>& dominoes) {
        map<pair<int, int>, int> m;
        int res = 0;
        for(auto x: dominoes) {
            if(x[0] > x[1]) {
                swap(x[0], x[1]);
            }
            m[{x[0], x[1]}]++;   
        }
        for(auto x: m) {
            res += x.second*(x.second - 1)/2;
        }
        return res;
    }
};
```

## [第二题 Shortest Path with Alternating Colors](https://leetcode.com/contest/weekly-contest-146/problems/shortest-path-with-alternating-colors/)

题目描述：给定一张又向图，节点的编号是0, 1, ..., n-1。在这张图中有两种边，一种是红色的一种是蓝色的。红色边和蓝色边要交替着走，求0到所有点的最短路径。

```
Input: n = 3, red_edges = [[0,1],[1,2]], blue_edges = []
Output: [0,1,-1]
解释：先走红色边，0-1，到node 1的最短距离为1。接下来走蓝色的边，发现没有蓝色的边，结束。

Input: n = 3, red_edges = [[0,1]], blue_edges = [[1,2]]
Output: [0,1,2]
解释：先走红色边，0-1，到node 1的最短距离为1。接下来走蓝色边，1-2，到node 2的最短距离为2。

Input: n = 3, red_edges = [[0,1],[0,2]], blue_edges = [[1,0]]
Output: [0,1,1]
解释：先走红色边，0-1, 到node 1的最短距离为1。接下来走蓝色边，发现0已经被访问过结束。
先走蓝色边，0-2，到node 2的最短距离为1。
```

思路：这道题的边是没有权重的，因此可以使用BFS寻找最短路径。在遍历的时候记录下下一次要走的颜色，不断更改。使用dist记录每个点到0的距离，dist是一个二维数组，dist[i][0]表示0到点i，当从i出发要走红色边的距离。dist[i][1]表示0到i，当从i出发要走蓝色边的距离。最终结果从两个值中取较小的那一个。在构造图的时候要把两种边都构造进去。遍历到每个点的时候，要找和从当前出发颜色相同的边。如果这个点，从当前颜色的相反颜色出发还没被遍历到，那么更新距离dist[j][1-c] = dist[i][c]+1。


```cpp
const int INF = 1e9 + 7;

class Solution {
public:
    vector<int> shortestAlternatingPaths(int n, vector<vector<int>>& red_edges, vector<vector<int>>& blue_edges) {
        vector<vector<int>> dist(n, vector<int>(2, INF));
        dist[0][0] = 0;
        dist[0][1] = 0;
        map<int, vector<pair<int, int>>> m;
        for(auto x: red_edges) {
            m[x[0]].push_back({x[1], 0});
        }
        for(auto x: blue_edges) {
            m[x[0]].push_back({x[1], 1});
        }
        queue<pair<int, int>> q;
        q.push({0, 0});
        q.push({0, 1});
        while(!q.empty()) {
            auto top = q.front(); q.pop();
            for(auto x: m[top.first]) {
                if(x.second == top.second && dist[x.first][1-top.second] == INF) {
                    dist[x.first][1-top.second] = dist[top.first][top.second] + 1;
                    q.push({x.first, 1-top.second});
                }
            }
        }
        vector<int> res;
        for(auto x: dist) {
            res.push_back(min(x[0], x[1]));
        }
        for(auto &x: res) {
            if(x == INF) x = -1;
        }
        return res;
    }
};
```

## [第三题 Minimum Cost Tree From Leaf Values](https://leetcode.com/contest/weekly-contest-146/problems/minimum-cost-tree-from-leaf-values/)

题目描述：给定一个数组，代表的是一棵树叶子节点中序遍历的结果。使用这些叶子节点构造一棵树，使得所有非叶子节点的加和最小。一个非叶子节点的值，等于他左子树和右字数的最大值的乘积。

```
Input: arr = [6,2,4]
Output: 32
Explanation:
There are two possible trees.  The first has non-leaf node sum 36, and the second has non-leaf node sum 32.

    24            24
   /  \          /  \
  12   4        6    8
 /  \               / \
6    2             2   4
```

思路：使用dp，dp[i][j]代表从叶子节点i到叶子节点j构成的树的最小和。对于i到j中的每一个叶子节点，不断分割。如果i==j代表现在在叶子节点，返回0。如果i+1 == j那么两个叶子节点构成一个新的节点，返回arr[i]*arr[j]。其他情况dp[i][j] = min(dp[i][k] + dp[k+1][j] + left * right)，其中left是从i到k的最大值，right是从k+1到j的最大值。

```cpp
const int INF = 1e9 + 7;

class Solution {
public:
    int dp[100][100];
    int mctFromLeafValues(vector<int>& arr) {
        memset(dp, -1, sizeof(dp));
        return dfs(arr, 0, arr.size()-1);
    }
    
    int dfs(vector<int> &arr, int start, int end) {
        if(start == end) return 0;
        if(start+1 == end) {
            dp[start][end] = arr[start]*arr[end];
            return dp[start][end];
        }
        if(dp[start][end] != -1) return dp[start][end];
        int ans = INF;
        for(int i=start; i<end; ++i) {
            int left=0, right=0;
            for(int j=start; j<=i;++j) left=max(left, arr[j]);
            for(int j=i+1; j<=end;++j) right=max(right, arr[j]);
            ans = min(ans, dfs(arr, start, i)+dfs(arr, i+1, end) + left*right);
        }
        dp[start][end] = ans;
        return ans;
    }
};
```