import time

matrix = [
[8, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 2, 0, 9, 0, 0, 0, 0],
[0, 0, 0, 2, 0, 0, 4, 0, 7],
[0, 0, 1, 0, 0, 0, 5, 0, 0],
[4, 0, 0, 0, 0, 7, 0, 0, 3],
[0, 0, 6, 0, 5, 0, 0, 8, 0],
[0, 7, 0, 0, 0, 9, 1, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 5, 0],
[0, 0, 0, 6, 8, 0, 0, 9, 0]]

result = [
[7, 5, 8, 1, 3, 4, 9, 2, 6],
[6, 9, 1, 2, 8, 0, 3, 0, 0],
[0, 0, 0, 0, 0, 6, 0, 0, 8],
[0, 8, 0, 9, 0, 0, 0, 0, 0],
[0, 3, 5, 0, 0, 0, 0, 0, 9],
[0, 0, 0, 0, 7, 2, 0, 4, 0],
[0, 0, 9, 5, 2, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 8, 6, 7],
[1, 0, 0, 3, 0, 0, 0, 0, 0]]

zero_cnt = 0
step = 0


def is_valid(x, y, value):
    x_block = x // 3 * 3
    y_block = y // 3 * 3

    for i in range(0, 9):
        if matrix[i][y] != 0 and i != x and matrix[i][y] == value:
            return False
    for j in range(0, 9):
        if matrix[x][j] != 0 and j != y and matrix[x][j] == value:
            return False
    for i in range(0, 3):
        for j in range(0, 3):
            if matrix[x_block+i][y_block+j] != 0 and (x_block+i != x or y_block+j != y) and matrix[x_block+i][y_block+j] == value:
                return False
    return True


def get_next_block():
    for i in range(0, 9):
        for j in range(0, 9):
            next_x, next_y = i, j
            if 0 <= next_x < 9 and 0 <= next_y < 9 and matrix[next_x][next_y] == 0:
                return next_x, next_y
    return -1, -1


def dfs():
    x, y = get_next_block()
    if x == -1 and y == -1:
        for m in matrix:
            print(m)
        return True
    # for m in matrix:
    #     print(m)
    # print('----{}, {} ----'.format(x, y))
    for num in range(1, 10):
        if is_valid(x, y, num):
            matrix[x][y] = num
            if dfs():
                return True
            matrix[x][y] = 0
    return False


t1 = time.time()
dfs()
t2 = time.time()
print(t2 - t1)

        
