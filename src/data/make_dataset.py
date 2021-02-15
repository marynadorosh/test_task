import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from textaugment import EDA
from nltk.tokenize import word_tokenize


class DataProcessing:

    def __init__(self, input_path, output_path):

        self.input_path = input_path
        self.output_path = output_path
        self.X = None
        self.label = None
        self.text = None


    def read_file(self):
        
        data = pd.read_csv(self.input_path, names=['text', 'label'])
        self.text = data.text
        if not data.label.isnull().all():
            self.label = data.label
        print(self.text.shape)


    def convert_to_vector(self, emb_dict):
        X = []
        emb_len = len([*emb_dict.values()][0])
        for sentence in self.text.values:
            vector = np.zeros((1, emb_len))
            words = [word for word in sentence.split() if word in emb_dict.keys()]
            if len(words):
                vector = np.mean([emb_dict[w] for w in words], axis=0)
            X.append(vector)
        self.X = np.vstack(X)


    def augment_text(self, def_val=3):
        eda = EDA()
        avg = int(len(self.label) / self.label.nunique())
        small_classes = (self.label.value_counts().reset_index(name='cnt')
                         .query(f'cnt < {avg}')['index'].values)

        for cl in tqdm(small_classes):
            tmp_df = self.text[self.label == cl]
 
            for sentence in tmp_df.values:

                text_aug = pd.Series([eda.synonym_replacement(sentence)
                                      for _ in range(def_val)])
                if sum(self.label==cl) > avg:
                    break
                self.text = self.text.append(text_aug, ignore_index=True)
                self.label = self.label.append(pd.Series([cl] * def_val),
                                               ignore_index=True)


    def shuffle_data(self):
        new_index = np.random.randint(len(self.label), size=len(self.label))
        self.label = self.label[new_index]
        self.text = self.text[new_index]


    def save_data(self):
        
        np.save(os.path.join(self.output_path, 'X.npy'), self.X)
        if self.label is not None:
            np.save(os.path.join(self.output_path, 'Y.npy'), 
                    self.label.to_numpy())


    @staticmethod
    def load_embedding(file_path):
        embedding_dict = {}
        with open(file_path, 'r') as f:
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                vectors = np.asarray(values[1:], 'float32')
                embedding_dict[word] = vectors
        f.close()
        return embedding_dict

