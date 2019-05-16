relations = set()
with open('data/re.csv', 'r') as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        splits = line.split(',')
        relations.add(splits[1])

for r in relations:
    print(r)
print(i)
print(len(relations))