# with open('README.md', 'r') as f:
#     lines = f.readlines()
#     cpp_flag = True
#     python_flag = True
#     python_cnt = 0
#     cpp_cnt = 0
#     start = 0
#     end = 0
#     for idx, line in enumerate(lines):
#         line = line.strip()
#         if "```python" in line:
#             start = idx
#             python_flag = False
#         if line == "```" and not python_flag:
#             end = idx
#             python_flag = True
#             python_cnt += end -start-1
#         if "```cpp" in line:
#             start = idx
#             cpp_flag = False
#         if line == "```" and not cpp_flag:
#             end = idx
#             cpp_flag = True
#             cpp_cnt += end-start-1
#     print("cpp_cnt = {}\npython_cnt = {}".format(cpp_cnt, python_cnt))

# n = input()
# l = n.split(' ')
# l = l[1:]
# l = [int(x) for x in l]
# min_v, max_v = l[0], l[0]
# res = l[0]
# for i in range(1, len(l)):
#     tx = max(l[i], max(l[i] * min_v, l[i] * max_v))
#     ti = min(l[i], min(l[i] * min_v, l[i] * max_v))
#     max_v = tx
#     min_v = ti
#     res = max(res, max_v)
# print(res)

# import math
# n = 7140229933
# prim = [3]
# q = math.sqrt(n)
# while prim[-1] < q:
#     prim.append(prim[-1]+2)
# for x in prim:
#     if n % x == 0:
#         print(x, n/x)
# print(83777*85229)

def isprime(n):
    if(n == 2 or n == 3):
        return True
    if(n%6!=1 and n%6!=5):
         return False
    n_sqrt = int(math.sqrt(n))
    for i in range(5, n_sqrt, 6):
        if n % i == 0 or n % (i+2) == 0:
            return False
    return True
    

import math
from tqdm import tqdm
start = 6541367000
end = 6541367999
q = int(math.sqrt(end)) + 2
q = q * 2
prim = [x for x in range(q)]
prims = []
for i in range(2, q):
    if prim[i] != 0:
        prims.append(i)
    for j in range(i+i, len(prim), i):
        prim[j] = 0
print(prims)
for n in range(start, end+1):
    for x in prims:
        m = n // x
        if n % x == 0 and m in prims:
            print(n, x, n//x)