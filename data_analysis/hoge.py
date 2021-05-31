l = [i for i in range(10)]
counter = 0
for _ in l:
    if l[0] == 5:
        break
    l.pop(0)
    counter+=1
print(l)
print(counter)