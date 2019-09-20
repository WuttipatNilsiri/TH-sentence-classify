import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import pandas as pd
import json

import tokenizer

import dummytokenize

import pickle

t = tokenizer.tokenizer()


class create_CNN():
    
    def __init__(self,X,Y,config,tokenizer=None):
        self.config_dict = json.load(open(config))
        self.lb = LabelEncoder()
        self.tfidf = TfidfVectorizer(
            tokenizer=tokenizer,
            preprocessor=tokenizer,
            token_pattern=None
        )

        self.x = X
        self.y = Y
    
    def fit_tfidf(self):
        print(self.x)
        self.fitted_matrix = self.tfidf.fit_transform(self.x).astype('float16')
        
        # saving tfidf matrix
        matrix = open(str(self.config_dict['path'] + self.config_dict['tfidf']), 'wb')
        pickle.dump(self.tfidf, matrix) 

        # return self.fitted_matrix
        
    def build_CNN(self):
        
        in_dim = self.fitted_matrix.shape[1]
        print("input Dim: " + str(in_dim) )
        out_dim = len(list(set(self.y)))
        print("output Dim: " + str(out_dim) )

        model = Sequential()
        model.add(Dense(3000, input_dim=in_dim, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1500, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(300, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(out_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        self.model = model

        return model
    
    def compile(self):

        encoded_y = self.lb.fit_transform(self.y)
        y_train = np_utils.to_categorical(encoded_y)

        # saving lb encoding matrix
        lb_encode = open(str(self.config_dict['path'] + self.config_dict['lb']) , 'wb')
        pickle.dump(self.lb, lb_encode)        

        clf = KerasClassifier(build_fn=self.build_CNN, epochs=15, batch_size=128)
        clf.fit(self.fitted_matrix,y_train)

        json_model = clf.model.to_json()
        
        # saving model arch
        open(str(self.config_dict['path'] + self.config_dict["model_arc"]), 'w').write(json_model)
        # saving weights
        clf.model.save_weights(str(self.config_dict['path'] + self.config_dict["model_w"]), overwrite=True)

        
        
        
#         x = clf.predict(transform(self.tfidf,'วนเพลงเดิมซ้ำอีก'))
#         print(x)
#         y_pred = self.lb.inverse_transform(x)
#         print(y_pred)


# def transform(matrix,sent):
#         print(sent)
#         _in = t.icutokenize(sent)
#         res = matrix.transform([_in]).astype('float16')

#         return res



        
        
        
        
    
        