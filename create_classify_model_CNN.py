import deepcut

from pythainlp.tokenize import word_tokenize

import pandas as pd

from sklearn.model_selection import train_test_split

from deeppavlov.core.data.simple_vocab import SimpleVocabulary

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

import tokenizer

import dummytokenize

import CNN

t = tokenizer.tokenizer()

def split_x_y(dataset,tokenize=False):
    
    data_x = []
    data_y = []
    
    for i in range(dataset.shape[0]):
    
        sent = dataset.iloc[i, 0]
        cat = dataset.iloc[i, 1]
        if tokenize:
            sent = t.icutokenize(sent)
        data_x.append(sent)
        data_y.append(cat)

    return data_x,data_y

class create_intent_classifier:
    
    def __init__(self, dataset_path, config):
        self.dataset_path = dataset_path
        self.config = config
    
    def create(self):
        dataset = pd.read_csv(self.dataset_path, encoding='UTF-8')
        data_x , data_y = split_x_y(dataset,tokenize=True)
        
        # Spilt train : test = 0.8 : 0.2
        # x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size = 0.8)
        # Full train (No Spliting)
        x_train = data_x
        y_train = data_y

        tokenizer = dummytokenize.dummytokenize()
        
        cnn = CNN.create_CNN(x_train,y_train,self.config,tokenizer=tokenizer.tokenize)

        cnn.fit_tfidf()
        cnn.build_CNN()
        cnn.compile()

creator = create_intent_classifier('./dataset/dataset.csv','./config.json')

creator.create()




    


    