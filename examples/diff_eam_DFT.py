import json


with open('structures.json', 'r') as f:
    structures = json.load(f)

for s in structures:
    lines = []
    tmp = ''
    for chr in s:
        tmp += chr
        if chr == '\n':
            lines.append(tmp)
            tmp = ''
    for line in lines:
        if 'Lattice' in line:
            print(line.split())
    quit()
