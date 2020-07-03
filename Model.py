from Dataset import NREDataset
import torch
import torch.nn as nn
from torch.nn import functional as F
class NREModel(nn.Module):
    def __init__(self, config):
        """
            args:
                config: some configurations of this model
        """
        super(NREModel, self).__init__()

        self.batch           = config.batch
        self.device          = config.device 
        self.embedding_size  = config.embedding_size
        self.embedding_dim   = config.embedding_dim

        self.hidden_dim      = config.hidden_dim
        self.tag_size        = config.tag_size
        


        self.pretrained      = config.pretrained

        if self.pretrained:
            self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(config.embedding), freeze=False)
        else:
            self.word_embeds = nn.Embedding(self.embedding_size,self.embedding_dim)
        
        self.hidden2tag = nn.Linear(in_features = self.hidden_dim, out_features = self.tag_size,  bias = True)

        #self.lstm = nn.LSTM(input_size = self.embedding_dim + self.pos_dim * 2, hidden_size=self.hidden_dim // 2,num_layers=1, bidirectional=True)
        self.gru = nn.GRU(input_size = self.embedding_dim, hidden_size=self.hidden_dim // 2, num_layers=1, bidirectional=True)
        

        self.dropout_emb=nn.Dropout(p=0.5)
        self.dropout_att=nn.Dropout(p=0.5)

        self.batchNorm = nn.BatchNorm1d(num_features = self.embedding_dim)
        self.hidden = self.init_hidden()


        self.att_weight = nn.Parameter(torch.randn(self.batch,1,self.hidden_dim))

    def init_hidden(self):
        return torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device)
        # https://pytorch.org/docs/stable/nn.html#lstm lstm的hidden和cell的形状
        # h_0 of shape(num_layers*num_direction, batch, hidden_size)
        # c_0 of shape(num_layers*num_direction, batch, hidden_size)
    def init_hidden_cell_lstm(self):
        return (torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device),
                torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device))

    def attention(self,H):
        M = torch.tanh(H) # # [batch, num_direction * hidden_size, seq_len]
        a = F.softmax(torch.bmm(self.att_weight,M),2)  # [bacth ,1, hidden_dim] * [batch, hidden_dim, seq_len]
        a = torch.transpose(a,1,2)                     # [batch, 1, seq_len] -> [batch, seq_len, 1]
        return torch.bmm(H,a)        # [batch, hidden_dim, seq_len] * [batch, seq_len, 1]->[batch, hidden_dim, 1]
    
    def forward(self,sentence):

        self.hidden = self.init_hidden()
        

        
        embeds = self.word_embeds(sentence)      # [batch, seq_len, embed_dim]
        embeds = torch.transpose(embeds, 1, 2)
        embeds = self.batchNorm(embeds)
        embeds = torch.transpose(embeds, 1, 2)
        embeds = self.dropout_emb(embeds)

        embeds = torch.transpose(embeds,0,1)    # 50 128 150   # [seq_len, batch, embed_dim]

        lstm_out, self.hidden = self.gru(embeds, self.hidden)  #[seq_len, batch, num_direction * hidden_size]


        lstm_out = torch.transpose(lstm_out,0,1) # [batch, seq_len, num_direction * hidden_size]
        lstm_out = torch.transpose(lstm_out,1,2) # [batch, num_direction * hidden_size, seq_len]
        
        

        att_out = torch.tanh(self.attention(lstm_out)) # [batch, num_direction * hidden_size, 1]
        att_out = self.dropout_att(att_out)            # [batch, num_direction * hidden_size, 1]
        
        
        att_out = torch.transpose(att_out, 1, 2) # [batch, 1, hidden_dim]
    
        res = self.hidden2tag(att_out) # [batch, 1, hidden_dim] * [batch, hidden_dim, tag_size]->[batch, 1, tag_size]
        res = torch.transpose(res, 1, 2)
        
        res = F.softmax(res,1)
        
        return res.view(self.batch,-1)
        