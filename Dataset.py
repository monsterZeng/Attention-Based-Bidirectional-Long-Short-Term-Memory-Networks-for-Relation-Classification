import torch
from Vocabulary import Vocabulary, SequenceVocabulary
from Vectorizer import NREVectorizer, np
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader

class NREDataset(Dataset):
    def __init__(self, news_df, vectorizer):
        """
        Args:
            news_df (pandas.DataFrame): the dataset
            vectorizer (NewsVectorizer): vectorizer instatiated from dataset
        """
        self.news_df = news_df
        self._vectorizer = vectorizer

        
        self._max_seq_length = 50 

        self.train_df = self.news_df[self.news_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.news_df[self.news_df.split=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.news_df[self.news_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}
        
        self.set_split('train')

        # Class weights
        class_counts = self.train_df.relation.value_counts().to_dict()
        def sort_key(item):
            return self._vectorizer.relation_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)


    @classmethod
    def load_dataset_and_make_vectorizer(cls, news_csv):
        """Load dataset and make a new vectorizer from scratch
                
            Args:
                surname_csv (str): location of the dataset
            Returns:
                an instance of SurnameDataset
        """
        news_df = pd.read_csv(news_csv)
        train_news_df = news_df[news_df.split=='train']
        return cls(news_df, NREVectorizer.from_dataframe(train_news_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, news_csv, vectorizer_filepath):
        """Load dataset and the corresponding vectorizer. 
        Used in the case in the vectorizer has been cached for re-use
            
        Args:
            news_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of NREDataset
        """
        news_df = pd.read_csv(news_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(news_df, vectorizer)

    @staticmethod  # https://blog.csdn.net/lihao21/article/details/79762681 实例方法/类方法/静态方法
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file
            
        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of SurnameVectorizer
        """
        with open(vectorizer_filepath, mode = "r", encoding = "utf-16") as fp:
            return NREVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json
            
        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w", encoding = "utf-16") as fp:
            json.dump(self._vectorizer.to_serializable(), fp, ensure_ascii = False)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer
        
    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
        
    def pos(self, num):
        if num < -40:
            return 0
        elif -40 <= num and num <= 40:
            return num + 40
        else:
            return 80

    def __len__(self):
        return self._target_size
        
    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
            
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        seq_vector = self._vectorizer.vectorize(row.sequence, self._max_seq_length)

        relation_index = self._vectorizer.relation_vocab.lookup_token(row.relation)
        
        
        return {'x_data': np.array(seq_vector, dtype = np.int64),
                'y_target': np.array(relation_index, dtype = np.int64),
                }
    
        
    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
            
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

