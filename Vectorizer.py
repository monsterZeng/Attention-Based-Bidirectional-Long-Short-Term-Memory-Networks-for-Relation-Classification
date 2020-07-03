from Vocabulary import Vocabulary, SequenceVocabulary
import jieba
import numpy as np

class NREVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""    
    def __init__(self, seq_vocab, relation_vocab):
        """
        Args:
            word_vocab (SequenceVocabulary): maps words to integers
            relation_vocab (Vocabulary): maps relation to integers
        """
        self.seq_vocab = seq_vocab
        self.relation_vocab = relation_vocab
    
    def vectorize(self, seq, vector_length = -1):
        """
        Args:
            seq (str): the string of words 
            vector_length (int): an argument for forcing the length of index vector
        Returns:
            the vetorized title (numpy.array)
        """
        data_list = list(jieba.cut(seq, cut_all = False))
        indices = [self.seq_vocab.lookup_token(token) for token in data_list]

        
        if vector_length < 0:
            vector_length = len(indices)
            
        out_vector = np.zeros(vector_length, dtype=np.int64)
        if len(indices) < 50:
            out_vector[:len(indices)] = indices
            out_vector[len(indices):] = (vector_length - len(indices)) * [self.seq_vocab.mask_index]
        else:
            out_vector = indices[:vector_length]
        return np.array(out_vector, dtype = np.float64)

    @classmethod
    def from_dataframe(cls, news_df):
        """Instantiate the vectorizer from the dataset dataframe
        
        Args:
            news_df (pandas.DataFrame): the target dataset
        Returns:
            an instance of the NREVectorizer
        """
        relation_vocab = Vocabulary()
        for relation in set(news_df.relation):
            relation_vocab.add_token(relation)
        
        seq_vocab = SequenceVocabulary()
        for sequence in news_df.sequence:
            word_list = list(jieba.cut(sequence, cut_all = False))
            seq_vocab.add_many(word_list)
        return cls(seq_vocab, relation_vocab)
        
    @classmethod
    def from_serializable(cls, contents):
        seq_vocab    =  SequenceVocabulary.from_serializable(contents['seq_vocab'])
        relation_vocab =  Vocabulary.from_serializable(contents['relation_vocab'])
        return cls(seq_vocab=seq_vocab, relation_vocab=relation_vocab)
    def to_serializable(self):
        return {'seq_vocab': self.seq_vocab.to_serializable(),
                'relation_vocab': self.relation_vocab.to_serializable()}
