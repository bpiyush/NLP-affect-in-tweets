import preProcess
import re
from nltk.tokenize import TweetTokenizer
import gensim
from gensim import corpora, models, similarities
import tweetPreprocessor

f_anger = open("./EI-reg-En-train/EI-reg-En-anger-train.txt")
angerTrain = f_anger.read()

f_fear = open("./EI-reg-En-train/EI-reg-En-fear-train.txt")
fearTrain = f_fear.read()

f_joy = open("./EI-reg-En-train/EI-reg-En-joy-train.txt")
joyTrain = f_joy.read()

f_sadness = open("./EI-reg-En-train/EI-reg-En-sadness-train.txt")
sadnessTrain = f_sadness.read()
# S_emotion is the set of all the tweets (actual) of emotion training set
[S_anger, y_anger] = preProcess.getData(angerTrain)
[S_fear, y_fear] = preProcess.getData(fearTrain)
[S_joy, y_joy] = preProcess.getData(joyTrain)
[S_sadness, y_sadness] = preProcess.getData(sadnessTrain)

corpus = [S_anger, S_fear, S_joy, S_sadness]

for z in corpus:
	for t in z:
		tweetPreprocessor.produceWordEmbd(t)
