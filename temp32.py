from nltk.tokenize import word_tokenize
import re
#----------------------NRC lexcions to be used---------------
path = 'NRC_lexicon/NRC-Sentiment-Emotion-Lexicons/'
subfolders = ['AutomaticallyGeneratedLexicons/', 'NRC-Emotion-Lexicon-v0.92/']

fol1 = ['NRC-Emoticon-AffLexNegLex-v1.0/', 'NRC-Emoticon-Lexicon-v1.0/', 'NRC-Hashtag-Emotion-Lexicon-v0.2/', 'NRC-Hashtag-Sentiment-AffLexNegLex-v1.0/', 'NRC-Hashtag-Sentiment-Lexicon-v1.0/']

f = open( 'NRC_lexicon/NRC-AffectIntensity-Lexicon.txt')
rawf = f.read()
g = open(path + subfolders[0] + fol1[2] + 'NRC-Hashtag-Emotion-Lexicon-v0.2.txt')
rawg = g.read()

p1 = re.compile(r"[a-z]+\t[?!~#a-z@\d_<>%*\\=/\^$&+_A-Z']+\t\d\.\d{5}")
m1 = p1.findall(rawg)

p2 = re.compile(r"[a-z]+\t\d\.\d{3}\t[a-z]+")
m2 = p2.findall(rawf)

word_list = []

def tokenize(string):
    l = len(string)
    idx = [0]
    for k  in range(l):
        if string[k] == '\t':
            idx.append(k)
    parts = [string[i:j] for i,j in zip(idx, idx[1:]+[None])]
    parts[1] = parts[1][1:]
    parts[2] = parts[2][1:]
    return parts
    pass

emotions = ['anger', 'joy', 'fear', 'anger']
for x in m1:
    t = tokenize(x)
    t[2] = float(t[2])
    word_list.append(t)

for x in m2:
    t = tokenize(x)
    t[1] = float(t[1])
    word_list.append([t[2], t[0], t[1]])


words = []
for z in word_list:
    words.append(z[0])
#print(words[0:50])
print(len(words))
#----------------DATA file handling------------------------------
import re
import numpy as np
import gensim.models as gsm
import os
import stop_words
import gensim
import tqdm
import time
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
import preProcess
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


folders = ["EI-reg-En", "2018-EI-reg-En", "2018-EI-reg-En"]
datatypes = ["train", "dev", "test"]
emotions = ["anger", "fear", "joy", "sadness"]

data = []
vocabulary = []

for i,x in enumerate(folders):
    for j,y in enumerate(emotions):
        f = open(x + "-" + datatypes[i] +"/" + x + "-" + y + "-" + datatypes[i] + ".txt")
        raw = f.read()
        g = preProcess.getData(raw)
        data.append(g)


train_anger_len = len(data[0][0])
train_fear_len = len(data[1][0])
train_joy_len = len(data[2][0])
train_sadness_len = len(data[3][0])

train_len = train_anger_len + train_fear_len + train_joy_len + train_sadness_len

dev_anger_len = len(data[4][0])
dev_fear_len = len(data[5][0])
dev_joy_len = len(data[6][0])
dev_sadness_len = len(data[7][0])

dev_len = dev_anger_len + dev_fear_len + dev_joy_len + dev_sadness_len

test_anger_len = len(data[8][0])
test_fear_len = len(data[9][0])
test_joy_len = len(data[10][0])
test_sadness_len = len(data[11][0])

test_len = test_anger_len + test_fear_len + test_joy_len + test_sadness_len

sm = 0
sum_arr = []
for j in range(len(data)):
    sm += len(data[j][0])
    sum_arr.append(sm)

print(sum_arr)

def preProcessTweet(rawTweet):
    tweet = rawTweet

    # Removing twitter handles' tags
    tweet = re.sub(r"@{1}[A-Za-z0-9_]+\s", ' ', tweet)

    # Removing web addresses
    tweet = re.sub(r"htt(p|ps)\S+", " ", tweet)

    # Removing email addresses
    emails = r'[a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}'
    tweet = re.sub(emails, " ", tweet)

    # Tokenizing based on whitespaces
    tokens = word_tokenize(tweet)
    # print(tokens)

    # Getting hashtags intact
    newTokens = []
    for i,x in enumerate(tokens):
        if x == '#' and i < len(tokens)-1:
            y = x + tokens[i+1]
            newTokens.append(y)
        else:
            if i>0:
                if (tokens[i-1]!='#'):
                    newTokens.append(x)
            else:
                newTokens.append(x)

    # Getting clitics intact
    finalTokens = []
    for j,x in enumerate(newTokens):
        S = ["'s", "'re", "'ve", "'d", "'m", "'em", "'ll", "n't"]
        if x in S:
            y = newTokens[j-1] + x
            finalTokens.append(y)
        else:
            if j<len(newTokens)-1:
                if newTokens[j+1] not in S:
                    finalTokens.append(x)
            else:
                finalTokens.append(x)

    # Eliminate case sensitivity
    for i,z in enumerate(finalTokens):
        finalTokens[i] = z.lower()

    return finalTokens
    pass

#--------------Generating vectors------------------------------
feat_vecs = np.zeros(train_len)
for i in tqdm.trange(4):
    a = data[i][0]
    for j,x in enumerate(a):
        y = 0
        h = preProcessTweet(x)
        for f in h:
            if f in words:
                idx = words.index(f)
                if word_list[idx][0] in emotions:
                    y += word_list[idx][2] 
        feat_vecs[j*bool(i==0) + sum_arr[i-1]*bool(i!=0)] += y

#print(feat_vecs[:1000])
np.save('word_intensity_feature', feat_vecs)


