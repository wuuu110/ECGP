from torchtext import data
import torch
import torch.nn.functional as F
from torchtext.vocab import GloVe
#class DataReader:
class get_train_test_loader:
    def __init__(self,base_path):
        super(get_train_test_loader,self).__init__()
        self.TEXT_FIELD = data.Field(tokenize='spacy')
        self.LABEL_FIELD = data.LabelField()
        self.train,self.test=data.TabularDataset.splits(path=base_path, train='train.csv', test='test.csv', format='csv',fields=[('Label', self.LABEL_FIELD),('Text', self.TEXT_FIELD)])

        #self.vectors = Vectors(name='./.vector_cache/glove.6B.300d.txt'
        self.TEXT_FIELD.build_vocab(self.train,  self.test,vectors=GloVe(name='6B', dim=100))
        self.LABEL_FIELD.build_vocab(self.train,self.test)
    def get_vocab(self):
        return self.TEXT_FIELD.vocab.vectors
    def get_vocab_size(self):
        return len(self.TEXT_FIELD.vocab)
    def get_num_classes(self):
        #print(self.LABEL_FIELD.vocab.itos)
        return len(self.LABEL_FIELD.vocab)
    def get_training_data(self):
        return self.train
    def get_testing_data(self):
        return self.test
    def get_token_tensor(self,text,MAX_SEQ_LEN):
        ret = torch.tensor([self.TEXT_FIELD.vocab.stoi[s] for s in text])
        ret = F.pad(ret, (0,MAX_SEQ_LEN-ret.shape[0]), mode='constant', value=0)
        return ret
    def get_label_id(self,label_text):
        return self.LABEL_FIELD.vocab.stoi[label_text[0]]


