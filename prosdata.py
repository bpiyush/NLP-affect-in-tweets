import preProcess
import re
from nltk.tokenize import TweetTokenizer
import gensim
from gensim import corpora, models, similarities

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
t = S_anger[len(S_anger)-2]
# print(t[len(t)-1].encode('unicode-escape'))

# print(S_anger[5][9].encode('unicode-escape'))


def matchEmoticons(DestFile):
	pattern = re.compile(u"\\\\U0001f[0-9][0-9A-Za-z][0-9A-Za-z]")
	match = pattern.findall(DestFile) 
	print(match)
	pass

encodedFile = angerTrain.encode('unicode-escape')
print(encodedFile) 
# matchEmoticons(encodedFile)
"""
def word2VecModel(corpus, emotion):
	tkn = TweetTokenizer()
	tokenizedCorpus = [tkn.tokenize(sent) for sent in corpus[emotion]]
	#tokenizedCorpus = [nltk.word_tokenize(sent.decode('utf-8')) for sent in corpus[emotion]]
	model = gensim.models.Word2Vec(tokenizedCorpus, min_count=1, size = 50, negative = 0, hs = 1)
	model.save('test_model')
	model = gensim.models.Word2Vec.load('test_model')
	# print(model)
	# print(tokenizedCorpus)
	print(model.most_similar('friend'))
"""