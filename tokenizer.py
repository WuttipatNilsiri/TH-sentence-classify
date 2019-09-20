import deepcut

from pythainlp.tokenize import word_tokenize

class tokenizer:

    def deepcuttokenize(self,sent):
        return list(filter(lambda a: a != '  ' and a != ' ' and a != '   ', deepcut.tokenize(sent)))
    
    def icutokenize(self,sent):
        return list(filter(lambda a: a != '  ' and a != ' ' and a != '   ', word_tokenize(sent, engine="icu")))

    def tokenize(self,sent):
        return list(filter(lambda a: a != '  ' and a != ' ' and a != '   ', word_tokenize(sent)))

