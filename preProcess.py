import re
import numpy as np
"""
f_anger = open("./EI-reg-En-train/EI-reg-En-anger-train.txt")
angerTrain = f_anger.read()

f_fear = open("./EI-reg-En-train/EI-reg-En-fear-train.txt")
fearTrain = f_fear.read()

f_joy = open("./EI-reg-En-train/EI-reg-En-joy-train.txt")
joyTrain = f_joy.read()

f_sadness = open("./EI-reg-En-train/EI-reg-En-sadness-train.txt")
sadnessTrain = f_sadness.read()

filetoPP = angerTrain
"""
def count_no_matches(match):
	count  = 0
	for x in match:
		count += 1
	return count

def getData(filetoPP):
	pIDs = re.compile(r"\d{4}-En-\d{5}")
	tweetIDs = pIDs.findall(filetoPP)

	p1 = re.compile(r"\d\.\d{3}")
	m1 = p1.findall(filetoPP)

	y_ideal = np.zeros((count_no_matches(m1), 1))

	for i,x in enumerate(m1):
		y_ideal[i] = float(x)

	pSent = re.compile(r"\d\t[\s\S].*?\t")
	mSent = pSent.findall(filetoPP)

	for i in range(len(mSent)):
		x = mSent[i]
		x = x[2:len(x)-1]
		mSent[i] = x

	print(len(y_ideal), len(mSent))
	return [mSent, y_ideal]
	pass

