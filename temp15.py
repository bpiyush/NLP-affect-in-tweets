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
from scipy.sparse import hstack

from correlation_pearson.code import CorrelationPearson

from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

from sklearn.decomposition import PCA


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
    #pca = PCA(n_components=4000)
    #pca.fit(unigram_training_words.todense())
    #t0 = pca.transform(unigram_training_words.todense())

    #print(unigram_training_words.shape)


    #Finding the tf-idf representation
    from sklearn.feature_extraction.text import TfidfTransformer 
    transformer=TfidfTransformer(norm=None,smooth_idf=False,sublinear_tf=False,use_idf=True)
    tfidf_training_words=transformer.fit_transform(unigram_training_words)-unigram_training_words
    #pca.fit(tfidf_training_words.todense())
    #t1 = pca.transform(tfidf_training_words.todense())
    #print (tfidf_training_words)


    # #Finding the bigram representation 
    bigram_vectorizer=CountVectorizer(ngram_range=(1,2))
    bigram_training_words=bigram_vectorizer.fit_transform(training_reviews)
    #pca.fit(bigram_training_words.todense())
    #t2 = pca.transform(bigram_training_words.todense())
    #print (bigram_training_words.shape)


    #Additional Representations
    #N-Gram
    ngram_vectorizer=CountVectorizer(ngram_range=(1,5))
    ngram_training_words=ngram_vectorizer.fit_transform(training_reviews)
    #pca.fit(ngram_training_words.todense())
    #t3 = pca.transform(ngram_training_words.todense())
    #print (ngram_training_words.shape)

    #Modified IDF
    transformer=TfidfTransformer(norm=None,smooth_idf=True,sublinear_tf=False,use_idf=True)
    modified_tfidf_training_words=transformer.fit_transform(unigram_training_words)
    #print (modified_tfidf_training_words)



    kf=KFold(n_splits=5)
    print(type(ngram_training_words), type(tfidf_training_words))
    #print(ngram_training_words.shape())

    regMethods = [ "Neural Nets", "Decision Tree", "Random Forests", "K-NN", "ADA-Boost", "Gradient-Boost"]

    regModels = [MLPRegressor(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(11, 8),  activation='logistic'),
    DecisionTreeRegressor(random_state=0), RandomForestRegressor(max_depth=2, random_state=0), KNeighborsRegressor(n_neighbors=2),
    AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0),
    GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, max_depth=3)
    ]
    def get_pos_half(z, y):
        for i,x in enumerate(y):
            if x < 0.5:
                z[i] = 0
        return z
        pass
    features = [unigram_training_words, bigram_training_words, ngram_training_words, tfidf_training_words]
    feature_names = ["unigram", "bigram", "ngram", "tfidf"]
    for z in range(6):
    
        for g in range(4):
            print("Using feature: "+ feature_names[g] + "*****************************************")
            finRes = []
            X = features[g]
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
                """
                y_test_new = get_pos_half(y_test, y_test)
                Rmodel_predicted_new = get_pos_half(Rmodel_predicted, y_test)
                y_test_new = y_test_new[y_test_new != 0]
                Rmodel_predicted_new = Rmodel_predicted_new[Rmodel_predicted_new != 0]
                """
                Res.append(c.result(y_test, Rmodel_predicted))
                print("Feature used: "+ feature_names[g] +"Regression model: "+regMethods[z] +"---- Pearson Coefficient for "+ emotions[i] + ": ", c.result(y_test, Rmodel_predicted))
            finRes.append(sum(Res)/5)
            print("--------------------------------------------")
            print("Avg PC for "+ emotions[i] + " with "+ feature_names[g] + " and "+ regMethods[z] ,sum(finRes))
            print("--------------------------------------------")
            avg_pc[z][i] += sum(finRes)
        """
        X_new = hstack([features[0], features[1]]).todense()
        Res2 = []
        print("Now working for bigram+unigram:++++++++++++++++++++")
        for train_index, test_index in kf.split(X_new):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X_new[train_index], X_new[test_index]
            y_train, y_test = y[train_index], y[test_index]
            Rmodel = regModels[z]
            Rmodel.fit(X_train, y_train)
            Rmodel_predicted = Rmodel.predict(X_test)
            y_test_new = get_pos_half(y_test, y_test)
            Rmodel_predicted_new = get_pos_half(Rmodel_predicted, y_test)
            y_test_new = y_test_new[y_test_new != 0]
            Rmodel_predicted_new = Rmodel_predicted_new[Rmodel_predicted_new != 0]
            Res2.append(c.result(y_test_new, Rmodel_predicted_new))
            print(regMethods[z] +"- Pearson Coefficient for "+ emotions[i] + ": ", c.result(y_test_new, Rmodel_predicted_new))
        """
        


print(avg_pc)
print(np.sum(avg_pc, axis=1)/4)





