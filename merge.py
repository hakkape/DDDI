import pandas as pd
import glob

frames = []

count = 0

for f in glob.glob('*.csv'):
    c = pd.read_csv(f)
    frames.append(c)
    count += 1
    print(count)

result = pd.concat(frames)
result.to_csv('join.csv', index=False)