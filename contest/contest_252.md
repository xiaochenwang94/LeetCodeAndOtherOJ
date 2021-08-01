# leetcode contest 252

***leetcode 1952 Three Divisors***

題目描述：给定一个整数n，如果n正好能被三个正整数整除，返回true，否则返回false。

思路：任何整数都能被1和自己整除，从2遍历到sqrt(n)，如果能够整除n的数字比1个多，那么就返回false。

```cpp
class Solution {
public:
    bool isThree(int n) {
        int x = floor(sqrt(n));
        int cnt = 0;
        for(int i=2;i<=x;++i) {
            if(n % i == 0) {
                cnt++;
                if (n / i != i) cnt++;
            }
            if(cnt > 1) return false;
        }
        return cnt == 1;
    }
};
```

***leetcode 1953 Maximum Number of Weeks for Which You Can Work***

题目描述：有n个项目，编号从0到n-1。milestones数组中存储了每个项目需要多少周完工。做工需要符合下面两个规则：
* 每周只能做一个项目，每周必须工作。
* 同一个项目不能连续做两周。

当所有项目都完成，或者只剩下了一个项目做了就会违反规定的时候，就停止工作。注意，在规定的限制下，有可能不能完成所有工作。求出在不违反规定的情况下最多能工作多少周。

```
Example 1:

Input: milestones = [1,2,3]
Output: 6
Explanation: One possible scenario is:
​​​​- During the 1st week, you will work on a milestone of project 0.
- During the 2nd week, you will work on a milestone of project 2.
- During the 3rd week, you will work on a milestone of project 1.
- During the 4th week, you will work on a milestone of project 2.
- During the 5th week, you will work on a milestone of project 1.
- During the 6th week, you will work on a milestone of project 2.
The total number of weeks is 6.
Example 2:

Input: milestones = [5,2,1]
Output: 7
Explanation: One possible scenario is:
- During the 1st week, you will work on a milestone of project 0.
- During the 2nd week, you will work on a milestone of project 1.
- During the 3rd week, you will work on a milestone of project 0.
- During the 4th week, you will work on a milestone of project 1.
- During the 5th week, you will work on a milestone of project 0.
- During the 6th week, you will work on a milestone of project 2.
- During the 7th week, you will work on a milestone of project 0.
The total number of weeks is 7.
Note that you cannot work on the last milestone of project 0 on 8th week because it would violate the rules.
Thus, one milestone in project 0 will remain unfinished.
 

Constraints:

n == milestones.length
1 <= n <= 105
1 <= milestones[i] <= 109
```

思路：如果所有的项目都可以做完，那么可以工作的周数为milestones每一项的和。能否做完所有项目，取决于最大项目需要的周数。如果最大的项目需要的周数比其他所有项目的和还多1周，那么就无法完成所有项目。能够做到的就是其他所有项目的和*2+1周。

```cpp
class Solution {
public:
    long long numberOfWeeks(vector<int>& milestones) {
        long long int sum = 0;
        long long int max_num = 0;
        for (auto x : milestones) {
            sum += x;
            max_num = max_num > x ? max_num : x;
        }
        if (max_num > (sum/2)) {
            return (sum-max_num)*2+1;
        }
        return sum;
    }
};
```

***leetcode 1954 Minimum Garden Perimeter to Collect Enough Apples***

题目描述：在一个2D的网格中，每个坐标点都有一颗苹果树。在（i，j）位置的苹果树，可以采集到|i|+|j|个苹果。你会从（0，0）为中心的位置，选择一个方形区域进行苹果的采集。给定一个整数neededApples表示需要采集的苹果数量，返回需要采集的区域的四个角的苹果数量和。


Example 1:
![](https://assets.leetcode.com/uploads/2019/08/30/1527_example_1_2.png)

```
Input: neededApples = 1
Output: 8
Explanation: A square plot of side length 1 does not contain any apples.
However, a square plot of side length 2 has 12 apples inside (as depicted in the image above).
The perimeter is 2 * 4 = 8.
Example 2:

Input: neededApples = 13
Output: 16
Example 3:

Input: neededApples = 1000000000
Output: 5040
```

思路：设x为正方形区域的半径。根据（0，0）将坐标轴上的点分成4部分。坐标轴上的苹果数量为：
$$sum_{axis}(x) = 4 * (1+x) * x / 2 = 2x^2+2x$$
根据坐标轴，将其他区域也分为4个部分（不含坐标轴），以右上角（第一象限）为例。

当x=1的时候，第一象限的苹果数量为：
$$sum_{part}(x=1) = 1+1=2$$
当x=2的时候，第一象限的苹果数量为：
$$sum_{part}(x=2) = sum_{part}(x=1) + ((x+1)+(x+x))*x/2*2-(x+x)$$
$$sum_{part}(x=2) = ssum_{part}(x=1) + 3x^2-x$$
如下图所示，绿框为$sum_{part}(x=1)$，那么$sum_{part}(x=2)$为$sum_{part}(x=1)$加上两个红框的值，再减去加重复的边角值。一个红框的值就是一个等差数列求和。：
![](pic/leetcode1954.png)

因此x和苹果总数的关系为：
$$sum_{total}(x) = sum_{axis}(x) + 4*sum_{part}(x)$$
因此，求出大于neededApples的x的最小值，返回x*8即可。

```cpp
class Solution {
public:
    long long minimumPerimeter(long long neededApples) {
        long long int total = 0;
        long long int part = 0;
        for(long long int i=1;i<=neededApples;++i) {
            total = 14 * i * i - 2*i +part;
            part = 12*i*i-4*i + part;
            if (total >= neededApples) return i*8;
        }
        return 0;
    }
};
```

***leetcode 1955 Count Number of Special Subsequences***

题目描述：nums数组仅由0，1，2组成。如果一个子序列是正整数个0，跟着正整数个1，再跟着正整数个2组成。那么这个子序列就是特殊的。例如：[0,1,2]和[0,0,1,1,1,2]就是特殊的。[2,1,0],[1],[0,1,2,0]就不是特殊的。给定一个数组nums，求出所有特殊的子序列个数，如果数字太大，返回模10^9 + 7的值。

```
Example 1:

Input: nums = [0,1,2,2]
Output: 3
Explanation: The special subsequences are [0,1,2,2], [0,1,2,2], and [0,1,2,2].
Example 2:

Input: nums = [2,2,0,0]
Output: 0
Explanation: There are no special subsequences in [2,2,0,0].
Example 3:

Input: nums = [0,1,2,0,1,2]
Output: 7
Explanation: The special subsequences are:
- [0,1,2,0,1,2]
- [0,1,2,0,1,2]
- [0,1,2,0,1,2]
- [0,1,2,0,1,2]
- [0,1,2,0,1,2]
- [0,1,2,0,1,2]
- [0,1,2,0,1,2]
 

Constraints:

1 <= nums.length <= 105
0 <= nums[i] <= 2
```

思路：dp[0]表示正整数个0为特殊子序列的个数，dp[1]表示正整数个0，跟着正整数个1为特殊子序列的个数。dp[2]表示正整数个0，跟着正整数个1，再跟着正整数个2为特殊子序列的个数。分下列三种情况讨论：
* 当nums[i] = 0，那么dp[0] = dp[0] + 1
* 当nums[i] = 1，那么dp[1] = dp[1] + dp[0]
* 当nums[i] = 2，那么dp[2] = dp[2] + dp[1]

```cpp
class Solution {
public:
    int countSpecialSubsequences(vector<int>& nums) {
        vector<long long int> dp {0, 0, 0};
        int mod = 1e9 + 7;
        for(auto x : nums) {
            dp[x] = ((dp[x] + dp[x]) % mod + (x > 0 ? dp[x-1] : 1)) % mod;
        }
        return dp[2];
    }
};
```