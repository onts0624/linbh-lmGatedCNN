# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:46:58 2019

@author: dell
"""
import time
import numpy as np
import collections
import os
from random import randint
def read_words(conf):
    words = []
    word_data=[]
    for file in os.listdir(conf.data_dir_seq):
        print(file)
        with open(os.path.join(conf.data_dir_seq, file), 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                tokens = line.split()
                if(len(tokens)<conf.max_seq_length):
                    words.extend((['<bos>']+tokens+['<eos>']))
                    word_data.append(['<bos>']+tokens+['<eos>'])
                # NOTE Currently, only sentences with a fixed size are chosen
                # to account for fixed convolutional layer size.
                #if len(tokens) == conf.context_size-2:
                    #words.extend((['<pad>']*(conf.filter_h/2)) + ['<s>'] + tokens + ['</s>'])
    return words,word_data

def index_words(words, conf):
    word_counter = collections.Counter(words).most_common(conf.vocab_size-2)
    word_to_idx = {'<pad>': 0,'<unk>': 1}
    idx_to_word = {0: '<pad>',1: '<unk>'}
    for i,_ in enumerate(word_counter):
        word_to_idx[_[0]] = i+2
        idx_to_word[i+2] = _[0]
    #data = []
    #for word in words:
    #    idx = word_to_idx.get(word)
    #    idx = idx if idx else word_to_idx['<unk>']
    #    print(idx)
    #    data.append(idx)
    return word_to_idx, idx_to_word

def identity(length):
    return np.identity(length)
#单位矩阵函数

def get_batch(x_batches, y_batches, batch_idx):
    x, y = x_batches[batch_idx], y_batches[batch_idx]
    batch_idx += 1
    if batch_idx >= len(x_batches):
        batch_idx = 0
    return x, y.reshape(-1,1), batch_idx

def data_to_index(index,data,conf,source):
    line_num=np.array(data)
    print(line_num.shape)
    data_ids=np.zeros((line_num.shape[0], conf.max_seq_length), dtype='int32')
    if(source=="seq"):
        der=conf.data_dir_seq
    else:
        der=conf.data_dir_label
    data_line_counter=0
    f1 = open('word_test.txt','w', encoding='UTF-8')
    for file in os.listdir(der):
        with open(os.path.join(der, file), 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                tokens = line.split()
                data_index_counter=0
                if(len(tokens)>conf.max_seq_length-1):
                    continue
                for word in tokens:
                    try:
                        f1.write(word+' ')
                        data_ids[data_line_counter][data_index_counter]=index[word]
                    except KeyError:
                        data_ids[data_line_counter][data_index_counter]=index['<unk>']
                    data_index_counter=data_index_counter+1
                data_line_counter=data_line_counter+1
                f1.write('\n')
            print("now loading line:"+str(data_line_counter))
    return data_ids

def prepare_data(conf):
    words,data_words = read_words(conf)
    word_to_idx, idx_to_word= index_words(words, conf)
    seq_ids=data_to_index(word_to_idx,data_words,conf,"seq")
    #labels的获得改为在train的时候通过tf对seq_id剪裁获得
    #labels_ids=tf.slice(seq_ids,[0,1],[-1,-1])#原来的句子为<B> A B C D E <E> 则对应需要被预测的序列就是A B C D E <E> 所以将原句子的<B>n拿掉就是对应向量
    return word_to_idx,idx_to_word,seq_ids
    

def getTrainBatch(seq_ids,conf):
    arr=np.zeros([conf.batch_size,conf.max_seq_length])
    for i in range(conf.batch_size):
            num=randint(2*conf.batch_size,seq_ids.shape[0])
            arr[i]=seq_ids[num-1:num]
    arr=arr.astype(np.int)
    return arr

def getTestBatch(labels_ids,seq_ids,conf):
    #labels=np.zeros([conf.batch_size,conf.max_seq_length])
    arr=np.zeros([conf.batch_size,conf.max_seq_length])
    for i in range(conf.batch_size):
            num=i+1
            #labels[i]=labels_ids[num-1:num]
            arr[i]=seq_ids[num-1:num]
    arr=arr.astype(np.int)
    #labels=labels.astype(np.int)
    #return arr,labels
    return arr

if __name__ == '__main__':
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    np.set_printoptions(threshold=np.inf)  
    flags = tf.app.flags
    #del_all_flags()
    flags.DEFINE_integer("vocab_size", 2000, "Maximum size of vocabulary")
    flags.DEFINE_integer("embedding_size", 200, "Embedding size of each token")
    flags.DEFINE_integer("max_seq_length",40, "Embedding size of each token")
    flags.DEFINE_integer("filter_size", 64, "Depth of each CNN layer")
    flags.DEFINE_integer("num_layers", 10, "Number of CNN layers")
    flags.DEFINE_integer("block_size", 5, "Size of each residual block")
    flags.DEFINE_integer("filter_h", 5, "Height of the CNN filter")
    flags.DEFINE_integer("context_size", 20, "Length of sentence/context")
    flags.DEFINE_integer("batch_size", 1, "Batch size of data while training")
    flags.DEFINE_integer("epochs", 50, "Number of epochs")
    flags.DEFINE_integer("num_sampled", 1, "Sampling value for NCE loss")
    flags.DEFINE_integer("learning_rate", 1.0, "Learning rate for training")
    flags.DEFINE_integer("momentum", 0.99, "Nestrov Momentum value")
    flags.DEFINE_integer("grad_clip", 0.1, "Gradient Clipping limit")
    flags.DEFINE_integer("num_batches", 0, "Predefined: to be calculated")
    flags.DEFINE_string("ckpt_path", "ckpt", "Path to store checkpoints")
    flags.DEFINE_string("summary_path", "logs", "Path to store summaries")
    #flags.DEFINE_string("data_dir", "data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled", "Path to store data")
    flags.DEFINE_string("data_dir_seq", "data/seq", "Path to store data of seq")
    flags.DEFINE_string("data_dir_label", "data/label", "Path to store data of label")
    flags.DEFINE_integer("iterations", 100000, "Number of iterations")
    print("flags complete")
    FLAGS = tf.flags.FLAGS
    word_to_idx,idx_to_word,seq_ids=prepare_data(FLAGS)
    print(word_to_idx)
    #for i in range(FLAGS.iterations):
      #  nextBatch,nextBatchLabels=getTrainBatch(labels_ids,seq_ids,FLAGS);
       # print(nextBatch)
       # print(nextBatchLabels)
    