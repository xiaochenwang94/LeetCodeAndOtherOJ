with open('README.md', 'r') as f:
    lines = f.readlines()
    cpp_flag = True
    python_flag = True
    python_cnt = 0
    cpp_cnt = 0
    start = 0
    end = 0
    for idx, line in enumerate(lines):
        line = line.strip()
        if "```python" in line:
            start = idx
            python_flag = False
        if line == "```" and not python_flag:
            end = idx
            python_flag = True
            python_cnt += end -start-1
        if "```cpp" in line:
            start = idx
            cpp_flag = False
        if line == "```" and not cpp_flag:
            end = idx
            cpp_flag = True
            cpp_cnt += end-start-1
    print("cpp_cnt = {}\npython_cnt = {}".format(cpp_cnt, python_cnt))
