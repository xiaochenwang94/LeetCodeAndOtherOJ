with open('README.md', 'r') as f:
    lines = f.readlines()
    flag = True
    cnt = 0
    start = 0
    end = 0
    for idx, line in enumerate(lines):
        if "```" in line:
            if flag:
                start = idx
                flag = False
            else:
                end = idx
                flag = True
                cnt += end-start-1
    print(cnt)
