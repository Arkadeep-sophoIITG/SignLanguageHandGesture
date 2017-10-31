import pandas as pd

df1 = pd.read_csv('samsung_2.csv')
df2 = pd.read_csv('SampleSolution.csv')
i=0
pos=-1
for e1 in df2['Filename']:
	pos=-1
	for e2 in df1['Filename']:
		if e2 == e1:
			pos=0
	if pos == -1:
		print (e1)

