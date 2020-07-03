from Dataset import NREDataset
from Model import NREModel
from argparse import Namespace
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import json
import jieba 
import torch
import codecs
from torch.utils.data import Dataset, DataLoader

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        

def make_embedding(word2vec_filepath, word_vocab, unkonw_embedding = None, embedding_dim = None):
    """
    args:
        word2vec_filepath(str):存储所有词嵌入的文件路径
        word_vocab(iterable container):此模型的词字典
        unkonw_embedding:未知词的词嵌入
    """
    word2vec = {}
    with open(word2vec_filepath, mode = "r", encoding = "utf-16") as fp:
        for line in fp.readlines():
            word2vec[line.split()[0]] = [float(data) for data in line.split()[1:]]
    
    if unkonw_embedding is None:
        unkonw_embedding = [1] * embedding_dim
    embedding = []
    embedding.append(unkonw_embedding) # embedding按照下标索引，第一个词为mask_embedding
    embedding.append(unkonw_embedding) # embedding按照下标索引，第二个词为unkonw_embedding
    
    for word in word_vocab:
        if word in word2vec:
            embedding.append(word2vec[word])
        else:
            embedding.append(unkonw_embedding)
    
    return embedding
  
def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}
    
def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"): 
    """
    A generator function which wraps the PyTorch DataLoader. It will 
    ensure each tensor is on the write device location.
    """

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict 
    
if __name__ == '__main__':
  # ---------------------------------------模型的参数------------------------------
  args = Namespace(
    # Data and path information
    data_csv         = "data/train_with_splits.csv",
    vectorizer_file  = "vectorizer.json",
    model_state_file = "model.pth",
    save_dir         = "model_storage/RE_classification",
    embedding_file   = "embedding.json",
    # Model hyper parameter
    hidden_dim       = 200,
    embedding_size   = None,
    tag_size         = None,
    embedding_dim    = 300,
    pos_size        = 82,  #不同数据集这里可能会报错。
    pos_dim         = 25,
    pretrained      = True,
    embedding       = None,
    # Training hyper parameter
    epochs = 100,
    learning_rate = 1e-3,
    batch = 128,
    seed=1337,
    early_stopping_criteria=5,

    # Runtime hyper parameter
    cuda=True,
    catch_keyboard_interrupt=True,
    reload_from_files=True,
    expand_filepaths_to_save_dir=True,
    )
  # ---------------------------------Check CUDA and create new folder---------------------
  if not torch.cuda.is_available():
    args.cuda = False

  args.device = torch.device("cuda" if args.cuda else "cpu")

  print("Using CUDA: {}".format(args.cuda))
  if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir,args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir,args.model_state_file)


  # Set seed for reproducibility
  set_seed_everywhere(args.seed, args.cuda)

  # handle dirs
  handle_dirs(args.save_dir)
  # ------------------------------------加载模型------------------------------------------
  if args.reload_from_files and os.path.exists(args.vectorizer_file):
    # training from a checkpoint
    dataset = NREDataset.load_dataset_and_load_vectorizer(args.data_csv, args.vectorizer_file)
  else:
    # create dataset and vectorizer
    dataset = NREDataset.load_dataset_and_make_vectorizer(args.data_csv)
    dataset.save_vectorizer(args.vectorizer_file)
  
  print("created dataset successfully")
      
  vectorizer = dataset.get_vectorizer()
  args.tag_size       = len(vectorizer.relation_vocab._token_to_idx)
  args.embedding_size = len(vectorizer.seq_vocab._token_to_idx)
  
  
  # ---------------------------------------加载词嵌入----------------------------
  embedding = []
  if args.pretrained:
    if os.path.exists(args.embedding_file):
      with open("embedding.json", mode = "r", encoding = "utf-8") as fp:
        embedding= json.load(fp)
      embedding = [eval(data) for word, data in embedding.items()]
      args.embedding = embedding
    else:
      # 如果没有预先制作好seq_vocab的词嵌入，那么就必须从一个新的word2vec文件来制作符合自身的embedding文件
      embedding = make_embedding(args.embedding_file, vectorizer.seq_vocab._token_to_idx)
  print("loaded embedding successfully")
  # --------------------------------------创建模型------------------------------
  model = NREModel(args).to(args.device)
  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
  dataset.class_weights = dataset.class_weights.to(args.device)
  criterion = nn.CrossEntropyLoss(dataset.class_weights)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)
  train_state = make_train_state(args)
  print("created model successfully and training model...")
  # --------------------------------------trian---------------------------------
  for epoch in range(args.epochs):
    # Iterate over training dataset

    # setup: batch generator, set loss and acc to 0, set train mode on
    # train_dataloader = D.DataLoader(train_datasets,args.batch,True, drop_last = True)
    print("epoch: ", epoch)
    model.train()
    dataset.set_split('train')
    batch_generator = generate_batches(dataset, batch_size=args.batch, device = args.device)

    running_loss = 0.0
    running_acc = 0.0
    for batch_index, batch_dict in enumerate(batch_generator):
        # the training routine is these 5 steps:

        # --------------------------------------    
        # step 1. zero the gradients
        optimizer.zero_grad()

        # step 2. compute the output
        sentence = torch.cuda.LongTensor(batch_dict['x_data'])

        tag = torch.cuda.LongTensor(batch_dict['y_target'])
        y = model(sentence)

        #tags = Variable(tag)
        loss = criterion(y, tag)      
        running_loss += (loss.item() - running_loss) / (batch_index + 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc_t = compute_accuracy(y, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state['train_loss'].append(running_loss)
    train_state['train_acc'].append(running_acc)
    print("train: ",running_acc, "%")
    
    dataset.set_split('val')
    batch_generator = generate_batches(dataset, batch_size=args.batch, device=args.device)
    running_loss = 0.
    running_acc = 0.
    model.eval()
    for batch_index, batch_dict in enumerate(batch_generator):

        sentence = torch.LongTensor(batch_dict['x_data']).to(args.device)
        tag = torch.LongTensor(batch_dict['y_target']).to(args.device)

        y = model(sentence)
        acc_t = compute_accuracy(y, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)
    
    train_state['val_loss'].append(running_loss)
    train_state['val_acc'].append(running_acc)
    
    scheduler.step(train_state['val_loss'][-1])
    print("eval: ",running_acc, "%")