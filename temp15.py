import pandas as pd
import numpy as np
import gensim
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Doc2Vec
import time
import tqdm
import os
import re
import preProcess
from nltk.util import ngrams

from correlation_pearson.code import CorrelationPearson

from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold


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

    for z in finalTokens:
        vocabulary.append(z)
    return ' '.join(finalTokens)
    pass

all_docs_list = []

for i in data:
    for j in range(len(i[0])):
        all_docs_list.append(i[0][j])

for i,x in enumerate(all_docs_list):
    all_docs_list[i] = preProcessTweet(x)

print("Length of vocabulary: ", len(vocabulary))
"""
training_reviews = data[0][0]
for i,x in enumerate(training_reviews):
    training_reviews[i] = preProcessTweet(x)
training_labels = data[0][1]
test_reviews = data[4][0]
for i,x in enumerate(test_reviews):
    test_reviews[i] = preProcessTweet(x)
test_labels = data[4][1]

#Finding the unigram representation
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
unigram_training_words=vectorizer.fit_transform(training_reviews)

#print(unigram_training_words.shape)


#Finding the tf-idf representation
from sklearn.feature_extraction.text import TfidfTransformer 
transformer=TfidfTransformer(norm=None,smooth_idf=False,sublinear_tf=False,use_idf=True)
tfidf_training_words=transformer.fit_transform(unigram_training_words)-unigram_training_words
#print (tfidf_training_words)


# #Finding the bigram representation 
bigram_vectorizer=CountVectorizer(ngram_range=(1,2))
bigram_training_words=bigram_vectorizer.fit_transform(training_reviews)
#print (bigram_training_words.shape)


#Additional Representations
#N-Gram
ngram_vectorizer=CountVectorizer(ngram_range=(1,5))
ngram_training_words=ngram_vectorizer.fit_transform(training_reviews)
#print (ngram_training_words.shape)

#Modified IDF
transformer=TfidfTransformer(norm=None,smooth_idf=True,sublinear_tf=False,use_idf=True)
modified_tfidf_training_words=transformer.fit_transform(unigram_training_words)
#print (modified_tfidf_training_words)


emotions = ['train_anger_', 'train_fear_', 'train_joy_', 'train_sadness_', 'dev_anger_', 'dev_fear_', 'dev_joy_', 'dev_sadness_']

kf=KFold(n_splits=5)

for training_index, test_index in kf.split(bigram_training_words):
    X_training, X_test = bigram_training_words[training_index], bigram_training_words[test_index]
    Y_training, Y_test = training_labels[training_index], training_labels[test_index]
    Y_training = Y_training.ravel()
    Y_test = Y_test.ravel()
    print("Training a Neural Network with ")
    mlp  = MLPRegressor(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(11, 8),  activation='logistic')
    mlp.fit(X_training, Y_training)
    mlp_predicted = mlp.predict(X_test)
    # print(dev_labels - mlp_predicted)
    c = CorrelationPearson()
    print("pearson-coefficient for "  ": ", c.result(Y_test, mlp_predicted))


regMethods = [ "Neural Nets", "Decision Tree", "Random Forests", "K-NN", "ADA-Boost", "Gradient-Boost"]
regModels = [MLPRegressor(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(11, 8),  activation='logistic'),
DecisionTreeRegressor(random_state=0), RandomForestRegressor(max_depth=2, random_state=0), KNeighborsRegressor(n_neighbors=2),
AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0),
GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, max_depth=3)
]

for z in tqdm.trange(6):
    finRes = []
    for i in range(1):
        X = bigram_training_words
        y = training_labels
        y = y.ravel()
        Res = []
        kf = KFold(n_splits=5)
        c = CorrelationPearson()
        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            Rmodel = regModels[z]
            Rmodel.fit(X_train, y_train)
            Rmodel_predicted = Rmodel.predict(X_test)
            Res.append(c.result(y_test, Rmodel_predicted))
            print(regMethods[z] +"- Pearson Coefficient for "+ emotions[i] + ": ", c.result(y_test, Rmodel_predicted))
        
        print(regMethods[z] + ":Avg of pearson-coefficients for the " + emotions[i] + " : ", sum(Res)/5)
        finRes.append(sum(Res)/5)
    print("--------------------------------------------")
    print("Final PC for "+ regMethods[z] ,sum(finRes))
    print("--------------------------------------------")
"""
emotions = ['train_anger_', 'train_fear_', 'train_joy_', 'train_sadness_', 'dev_anger_', 'dev_fear_', 'dev_joy_', 'dev_sadness_']


print("Using tfidf_training_words---------------------------------------------")

avg_pc = np.zeros((6,4))
for i in tqdm.trange(4):
    RS = []
    training_reviews = data[i][0]
    for j,x in enumerate(training_reviews):
        training_reviews[j] = preProcessTweet(x)
    training_labels = data[i][1]

    #Finding the unigram representation
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer=CountVectorizer()
    unigram_training_words=vectorizer.fit_transform(training_reviews)

    #print(unigram_training_words.shape)


    #Finding the tf-idf representation
    from sklearn.feature_extraction.text import TfidfTransformer 
    transformer=TfidfTransformer(norm=None,smooth_idf=False,sublinear_tf=False,use_idf=True)
    tfidf_training_words=transformer.fit_transform(unigram_training_words)-unigram_training_words
    #print (tfidf_training_words)


    # #Finding the bigram representation 
    bigram_vectorizer=CountVectorizer(ngram_range=(1,2))
    bigram_training_words=bigram_vectorizer.fit_transform(training_reviews)
    #print (bigram_training_words.shape)


    #Additional Representations
    #N-Gram
    ngram_vectorizer=CountVectorizer(ngram_range=(1,5))
    ngram_training_words=ngram_vectorizer.fit_transform(training_reviews)
    #print (ngram_training_words.shape)

    #Modified IDF
    transformer=TfidfTransformer(norm=None,smooth_idf=True,sublinear_tf=False,use_idf=True)
    modified_tfidf_training_words=transformer.fit_transform(unigram_training_words)
    #print (modified_tfidf_training_words)



    kf=KFold(n_splits=5)


    regMethods = [ "Neural Nets", "Decision Tree", "Random Forests", "K-NN", "ADA-Boost", "Gradient-Boost"]

    regModels = [MLPRegressor(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(11, 8),  activation='logistic'),
    DecisionTreeRegressor(random_state=0), RandomForestRegressor(max_depth=2, random_state=0), KNeighborsRegressor(n_neighbors=2),
    AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0),
    GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, max_depth=3)
    ]

    for z in range(6):
        finRes = []
        X = ngram_training_words
        y = training_labels
        y = y.ravel()
        Res = []
        kf = KFold(n_splits=5)
        c = CorrelationPearson()
        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            Rmodel = regModels[z]
            Rmodel.fit(X_train, y_train)
            Rmodel_predicted = Rmodel.predict(X_test)
            Res.append(c.result(y_test, Rmodel_predicted))
            print(regMethods[z] +"- Pearson Coefficient for "+ emotions[i] + ": ", c.result(y_test, Rmodel_predicted))
        
        finRes.append(sum(Res)/5)
        print("--------------------------------------------")
        print("Avg PC for "+ emotions[i] + " with " + regMethods[z] ,sum(finRes))
        print("--------------------------------------------")
        avg_pc[z][i] += sum(finRes)


print(avg_pc)
print(np.sum(avg_pc, axis=1)/4)





