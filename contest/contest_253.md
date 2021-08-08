# contest 253

***leetcode 1961 Check If String Is a Prefix of Array***

题目描述：给定一个字符串和一个但是数组，判断能否使用数组中前k个单词拼接成字符串。

```cpp
class Solution {
public:
    bool isPrefixString(string s, vector<string>& words) {
        int s_len = s.size();
        string cur="";
        for(auto x: words) {
            cur = cur + x;
            if(cur.size() > s_len) {
                return false;
            } 
            if(cur == s) {
                return true;
            }
        }
        return false;
    }
};
```


***leetcode 1962 Remove Stones to Minimize the Total***

题目描述：piles表示n堆石头，每次可以挑一堆石头，移除floor(piles[i]/2)块。求移除k次之后最少还剩多少块。

思路：使用优先队列，每次找到最多的一组进行移除。


```cpp
class Solution {
public:
    int minStoneSum(vector<int>& piles, int k) {
        priority_queue<int> stones;
        long long int total = 0;
        for(auto x : piles) {
            stones.push(x);
            total += x;
        }
        
        for(int i=0;i<k;++i) {
            int stone = stones.top();
            stones.pop();
            int remain = (stone + 1) / 2;
            total = total - (stone - remain);
            stones.push(remain);
        }
        return total;
    }
};
```

***leetcode 1963 Minimum Number of Swaps to Make the String Balanced***

题目描述：给定一个由“[”和“]”组成的字符串，调整顺序使得括号序列合法。合法是指：
* 空字符串合法
* AB合法，如果A合法并且B合法
* [C]合法

可以调换字符串中任意两个字符的位置，求最少调换多少次可以让字符串合法。

思路：合法的部分不用调换，可以先将合法的部分去除掉，这样得到的字符串就是]]]][[[[。调换的次数等于((size/2)+1)/2

```cpp
class Solution {
public:
    int minSwaps(string s) {
        stack<char>  st;
        for(auto x: s) {
            if(!st.empty() && st.top() == '[' && x == ']') {
                st.pop();
            } else {
                st.push(x);
            }
        }
        return (st.size() / 2 + 1) / 2;
    }
};
```





***leetcode 1964 Find the Longest Valid Obstacle Course at Each Position***

题目描述：翻译过来就是找到每一个位置前面的最长递增子序列（非严格）的长度。

思路：dp的思路，使用dp数组记录前面每一个位置的最长递增子序列的长度。对于位置i遍历0到i-1，对于每一个nums[j] < nums[i]的位置比较dp[j] + 1和dp[i]的的大小，不断更新。时间复杂度为O(n^2)。这种做法最终超时。

```cpp
class Solution {
public:
    vector<int> longestObstacleCourseAtEachPosition(vector<int>& obstacles) {
        int size = obstacles.size();
        vector<int> dp(size, 1);
        for(int i=0;i<size;++i) {
            for(int j=0;j<i;++j) {
                if(obstacles[i] >= obstacles[j]) {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
        }
        return dp;
    }
};
```

思路二：转换为第300题，第二种解题思路。

```下面我们来看一种优化时间复杂度到O(nlgn)的解法，这里用到了二分查找法，所以才能加快运行时间哇。思路是，我们先建立一个数组ends，把首元素放进去，然后比较之后的元素，如果遍历到的新元素比ends数组中的首元素小的话，替换首元素为此新元素，如果遍历到的新元素比ends数组中的末尾元素还大的话，将此新元素添加到ends数组末尾(注意不覆盖原末尾元素)。如果遍历到的新元素比ends数组首元素大，比尾元素小时，此时用二分查找法找到第一个不小于此新元素的位置，覆盖掉位置的原来的数字，以此类推直至遍历完整个nums数组，此时ends数组的长度就是我们要求的LIS的长度，特别注意的是ends数组的值可能不是一个真实的LIS，比如若输入数组nums为{4, 2， 4， 5， 3， 7}，那么算完后的ends数组为{2， 3， 5， 7}，可以发现它不是一个原数组的LIS，只是长度相等而已，千万要注意这点。```

使用一个mono数组，表示上面的ends。每次使用upper bound找到mono中第一个大于nums[i]的数字，如果找到了，那么替换nums[i]，否则再最后push back。对于nums[i]再mono中的位置就是最长递增子序列的长度。这样的时间负责度是O(log n)。

```cpp
class Solution {
public:
    vector<int> longestObstacleCourseAtEachPosition(vector<int>& obstacles) {
        int size = obstacles.size();
        vector<int> mono, res;
        for(int i=0;i<size;++i) {
            auto iter = upper_bound(mono.begin(), mono.end(), obstacles[i]);
            if(iter != mono.end()) {
                *iter = obstacles[i];
                res.push_back((iter-mono.begin()) + 1);
            } else {
                mono.push_back(obstacles[i]);
                res.push_back(mono.size());
            }
        }
        return res;
    }
};
```

