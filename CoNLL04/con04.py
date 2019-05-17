nerl = set()
re = set()
with open('CoNLL04/train.txt', 'r') as f:
    for line in f:
        if line.startswith('#doc') is False:
            splits = line.split('\t')
            nerl.add(splits[2])
            r = splits[3][1:-1]
            s = r.split(', ')  # some word has more than one relation
            for i in range(len(s)):
                s[i] = s[i][1:-1]  # eliminate "
            for ss in s:
                re.add(ss)

print(nerl)
print(re)