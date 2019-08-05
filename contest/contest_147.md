# Leetcode Contest 147 题解
## [第一题 N-th Tribonacci Number](https://leetcode.com/contest/weekly-contest-147/problems/n-th-tribonacci-number/)
题目描述：Tribonacci 序列的定义如下:

T0 = 0, T1 = 1, T2 = 1, and Tn+3 = Tn + Tn+1 + Tn+2 for n >= 0.

输入n，输出Tn

```
Input: n = 4
Output: 4
Explanation:
T_3 = 0 + 1 + 1 = 2
T_4 = 1 + 1 + 2 = 4
```

思路：按照定义求即可

```cpp
class Solution {
public:
    int tribonacci(int n) {
        if(n==0) return 0;
        if(n==1 || n==2) return 1;
        int n0=0, n1=1, n2=1;
        int res = 0;
        for(int i=0;i<n-2;++i) {
            res = n0+n1+n2;
            n0 = n1;
            n1 = n2;
            n2= res;
        }
        return res;
    }
};
```

## [第二题 Alphabet Board Path](https://leetcode.com/contest/weekly-contest-147/problems/alphabet-board-path/)

题目描述：有一个棋盘如下：
board = ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"]，开始在棋盘的[0][0]位置。在棋盘上有如下几种操作：

* U：向上移动一格
* D：向下移动一格
* L：向左移动一格
* R: 向右移动一格
* !: 把当前位置的字符放到结果中

输入一个字符串s，使用最少的移动次数，从棋盘上找出s。

```
Input: target = "leet"
Output: "DDR!UURRR!!DDD!"

Input: target = "code"
Output: "RR!DDRR!UUL!R!"
```

思路：因为棋盘是按照字母顺序排放的，我们可以直接计算出从当前位置，到下一个字母的x方向距离和y方向距离。两点之间直线最短，我们只要不绕圈，沿着坐标轴移动x和y，肯定是最少的移动次数。计算target[i-1]和target[i]的距离要把x轴和y轴分开计算，对于target[i]来说，x=target[i] / 5, y=target[i] % 5。在x轴和y轴要移动的距离为x_delta=abs(x_prev-x)， y_delta=abs(y_prev-y)。一个坑在最后一行，我们要分情况讨论。1) 如果上一个字母是z，当前要去的字母不是z，那么要先向上移动，再向右移动。2) 如果上一个字母不是z，当前要找z，那么要先想左移动，再向下移动。

```cpp
class Solution {
public:
    string alphabetBoardPath(string target) {
        char prev='a'-'a';
        string res = "";
        for(int i=0;i<target.size();++i) {
            target[i]-='a';
            int prev_r = prev / 5;
            int prev_c = prev % 5;
            int cur_r = target[i] / 5;
            int cur_c = target[i] % 5;
            // cout << target[i] << " " << prev << " " << prev_r << " " << prev_c << " " << cur_r << " " << cur_c << endl; 
            char r = cur_r > prev_r ? 'D' : 'U';
            char c = cur_c > prev_c ? 'R' : 'L';
            if(target[i] == (int)('z'-'a')) {
                for(int j=0;j<abs(prev_c-cur_c);++j) {
                    res+=c;
                }
                for(int j=0;j<abs(prev_r-cur_r);++j) {
                    res+=r;
                }
                
            } else if(prev == (int)('z'-'a')) {
                for(int j=0;j<abs(prev_r-cur_r);++j) {
                    res+=r;
                }
                for(int j=0;j<abs(prev_c-cur_c);++j) {
                    res+=c;
                }
                
            } else {
                for(int j=0;j<abs(prev_r-cur_r);++j) {
                    res+=r;
                }
                for(int j=0;j<abs(prev_c-cur_c);++j) {
                    res+=c;
                }
            }
            res+="!";
            prev = target[i];
        }
        return res;
    }
};
```

参考Leetcode neal_wu的做法才发现，只要调整移动方向的顺序是不需要特殊考虑z的。因为每个字母的左边和上边的字母一定是存在的，那么只要先把左和上移动了，就不会发生z移动出边界的情况。

```cpp
class Solution {
public:
    string alphabetBoardPath(string target) {
        int r = 0, c = 0;
        string moves = "";

        for (char letter : target) {
            int row = (letter - 'a') / 5, col = (letter - 'a') % 5;

            while (c > col) {
                moves += 'L';
                c--;
            }

            while (r > row) {
                moves += 'U';
                r--;
            }

            while (r < row) {
                moves += 'D';
                r++;
            }

            while (c < col) {
                moves += 'R';
                c++;
            }

            moves += '!';
        }

        return moves;
    }
};
```

## [第三题 Largest 1-Bordered Square](https://leetcode.com/contest/weekly-contest-147/problems/largest-1-bordered-square/)

题目描述：输入一个二维的0/1矩阵，求出最大的由1围成的正方形的面积，如果不存在返回0。

```
Input: grid = [[1,1,1],[1,0,1],[1,1,1]]
Output: 9

Input: grid = [[1,1,0,0]]
Output: 1
```

思路：对于每一个点grid[i][j]，如果是1，以i,j为正方形的左上角，边长为res+1，找存不存在这样的正方形。如果存在更新res，如果不存在，一直找到边界为止。

```cpp
class Solution {
public:
    int largest1BorderedSquare(vector<vector<int>>& grid) {
        int res = 0;
        for(int i=0;i<grid.size();++i) {
            for(int j=0;j<grid[0].size();++j) {
                if(grid[i][j] == 0) continue;
                for(int len=max(res,1);i+len<=grid.size()&&j+len<=grid[0].size();++len) {
                    // col right
                    bool flag = true;
                    for(int r=0;r<len;++r) {
                        if(grid[i][j+r] == 0) {
                            flag = false;
                            break;
                        }
                    }
                    if(!flag) break;
                    for(int r=0;r<len;++r) {
                        if(grid[i+r][j] == 0) {
                            flag = false;
                            break;
                        }
                    }
                    if(!flag) break;
                    for(int r=0;r<len;++r) {
                        if(grid[i+len-1][j+r] == 0) {
                            flag = false;
                            break;
                        }
                    }
                    if(!flag) continue;
                    for(int r=0;r<len;++r) {
                        if(grid[i+r][j+len-1] == 0) {
                            flag = false;
                            break;
                        }
                    }
                    if(!flag) continue;
                    res = max(res, len);
                }
            }
        }
        return res*res;
    }
};
```

## [第四题 Stone Game II](https://leetcode.com/contest/weekly-contest-147/problems/stone-game-ii/)
题目描述：给定一个数组piles，代表很多堆石子。piles[i]表示第i堆石子有多少个。游戏中A和B两个人轮流取石子，从左到右按顺序取。每次可以取走的堆数为X，其中1 <=x <= M, M=max(x, M)。开始的时候M为1。假设A和B都是很聪明的，求A最多能取多少个石子。

```
Input: piles = [2,7,9,4,4]
Output: 10
Explanation:  If Alex takes one pile at the beginning, Lee takes two piles, then Alex takes 2 piles again. Alex can get 2 + 4 + 4 = 10 piles in total. If Alex takes two piles at the beginning, then Lee can take all three piles left. In this case, Alex get 2 + 7 = 9 piles in total. So we return 10 since it's larger. 
```

思路：这道题是一个博弈的过程，也就是一个极大极小搜索的过程。目标函数定义为A要尽可能的比B取的多，为了表示两个人轮流的过程，我们使用当前结果减去后面搜索的结果。这样相当于一个人正，一个人负，每次都最大化自己，同时相当于最小化对方。

参考Leetcode neal_wu的代码

```cpp
const int INF = 1e9+7;

class Solution {
public:
    int n;
    vector<int> piles_;
    map<pair<int, int>, int> m;
    int solve(int start, int M) {
        if(start >= n) return 0;
        if(m.find({start, M}) != m.end())
            return m[{start, M}];
        int ans = -INF;
        int sum = 0;
        for(int x=1; x<=2*M && start+x<=n; ++x) {
            sum += piles_[start+x-1];
            ans = max(ans, sum-solve(start+x, max(M, x)));            
        }
        m[{start, M}] = ans;
        return ans;
    }
    
    int stoneGameII(vector<int>& piles) {
        n = piles.size();
        piles_ = piles;
        int total = accumulate(piles.begin(), piles.end(), 0);
        return (solve(0, 1)+total) / 2;
    }
};
```