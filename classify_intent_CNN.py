import pickle

import tokenizer

import dummytokenize

from keras.models import model_from_json

t = tokenizer.tokenizer()

import json

class classifier:
    
    def __init__(self,config):

        self.config_dict = json.load(open(config))

        self.matrix = pickle.load(open(str(self.config_dict["path"]+self.config_dict["tfidf"]), 'rb'))

        self.model = model_from_json(open(str(self.config_dict["path"]+self.config_dict["model_arc"])).read())
        self.model.load_weights(str(self.config_dict["path"]+self.config_dict["model_w"]))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

        # self.model = pickle.load(open(model_path, 'rb'))
        self.lb = pickle.load(open(str(self.config_dict["path"]+self.config_dict["lb"]), 'rb'))

    def classify(self,sent):
        print(sent)
        _in = t.icutokenize(sent)
        res = self.matrix.transform([_in]).astype('float16')
        x = self.model.predict_classes(res)
        y_pred = self.lb.inverse_transform(x)
        return y_pred

classifier = classifier('./config.json')

print( classifier.classify("้ลบเพลงหมดลิสท์"))