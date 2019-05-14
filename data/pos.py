total_tags = set()
with open('data/train.txt', 'r') as f:
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            continue
        splits = line.split(' ')
        total_tags.add(splits[-1][:-1])

print(total_tags)
print(len(total_tags))