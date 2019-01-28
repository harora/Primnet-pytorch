import os


def getdict(f):
	f=open(f)
	li=[]
	for i in range(2109):
		l=f.readline()
		li.append(l.split(' ')[3])
	li=list(set(li))
	for i in range(len(li)):
		print "'",li[i].strip(),"':",i,','
file='chimface/chimpface/data_CZoo/annotations_czoo.txt'
# getdict(file)

import numpy as np

# i=np.load('chimface/chimpface/data_CZoo/ctai_zoo_data_split_5.npz')
# print i.keys()
# i1=i['xte']
# i2=i['yte']
# print i1.shape
# lines=[]
# for i in range(1457):
# 	lines.append(i1[i]+' '+str(i2[i])+'\n')

# f=open('chimface/chimpface/split5test.txt','w')
# for l in lines:
# 	f.write(l)
# f.close()


ref1='chimface/chimpface/data_CZoo/annotations_czoo.txt'
ref2='chimface/chimpface/data_CTai/annotations_ctai.txt'
list1='chimface/chimpface/split1test.txt'
list2='chimface/chimpface/lists/split1test.txt'




dic1={}
re1=open(ref1)
for i in range(2109):
	lin=re1.readline()
	lin1=lin.split(' ')[1].split('/')[1]
	dic1[lin1]=lin


dic2={}
re2=open(ref2)
for i in range(5078):
	lin=re2.readline()
	lin1=lin.split(' ')[1].split('/')[1]
	dic2[lin1]=lin

l1=open(list1)
l2=open(list2,'w')
for i in range(6030):
	lin=l1.readline()
	linx=lin.split(' ')[0]
	lin1=linx.split('/')[0]
	lin2=linx.split('/')[2]
	print lin2
	if lin1=='data_CZoo':
		l2.write(lin.strip() + ' '+ ' '.join(dic1[lin2].split(' ')[10:19])+'\n')
	elif lin1=='data_CTai':
		l2.write(lin.strip() + ' '+ ' '.join(dic2[lin2].split(' ')[10:19])+'\n')
	
