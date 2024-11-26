import os, random


valid_samples = []
with open('test.txt', 'r', encoding='utf-8') as fin:
    for line in fin:
        valid_samples.append(line.strip())

random.shuffle(valid_samples)
valid_samples = valid_samples[:len(valid_samples) // 3]

with open('valid.txt', 'w', encoding='utf-8') as fout:
    for sample in valid_samples:
        fout.write(sample + '\n')