from __future__ import unicode_literals

import dynet_config
#dynet_config.set(random_seed=127, autobatch=1)

import io
import re
import sys
import time
import random
import pickle

from argparse import ArgumentParser
from collections import Counter, defaultdict
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report as cr

import gensim
import dynet as dy
import numpy as np

class Meta:
    def __init__(self):
        self.c_dim = 32  # character-rnn input dimension
        self.w_dim_e = 0  # pretrained word embedding size (0 if no pretrained embeddings)
        self.n_hidden = 128  # pos-mlp hidden layer dimension
        self.lstm_char_dim = 64  # char-LSTM output dimension
        self.lstm_word_dim = 128  # LSTM (word-char concatenated input) output dimension


class Tagger():
    def __init__(self, model=None, meta=None, wvm=None):
        self.model = dy.Model()
        self.meta = pickle.load(open('%s.meta' %model, 'rb')) if model else meta
        self.trainer = self.meta.trainer(self.model)
        # pretrained embeddings
        if wvm:
            self.wvm = wvm
            self.meta.w_dim_e = wvm.syn0.shape[1]

        # MLP on top of biLSTM outputs 100 -> 32 -> ntags
        self.w1 = self.model.add_parameters((self.meta.n_hidden, self.meta.lstm_word_dim*2))
        self.w2 = self.model.add_parameters((self.meta.n_tags, self.meta.n_hidden))
        self.b1 = self.model.add_parameters(self.meta.n_hidden)
        self.b2 = self.model.add_parameters(self.meta.n_tags)

        self.aw = self.model.add_parameters((self.meta.lstm_word_dim*2, self.meta.lstm_word_dim*2))
        self.ab = self.model.add_parameters(self.meta.lstm_word_dim*2)
        self.av = self.model.add_parameters((1, self.meta.lstm_word_dim*2))

        # word-level LSTMs
        self.fwdRNN = dy.LSTMBuilder(1, self.meta.w_dim_e+self.meta.lstm_char_dim*0, self.meta.lstm_word_dim, self.model) 
        self.bwdRNN = dy.LSTMBuilder(1, self.meta.w_dim_e+self.meta.lstm_char_dim*0, self.meta.lstm_word_dim, self.model)
        self.fwdRNN2 = dy.LSTMBuilder(1, self.meta.lstm_word_dim*2, self.meta.lstm_word_dim, self.model) 
        self.bwdRNN2 = dy.LSTMBuilder(1, self.meta.lstm_word_dim*2, self.meta.lstm_word_dim, self.model)

        # char-level LSTMs
        self.cfwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)
        self.cbwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)
        # unk for unknown word embeddings
        self.unk = np.zeros(self.meta.w_dim_e)

        # define char lookup table
        self.CHAR_LOOKUP = self.model.add_lookup_parameters((self.meta.n_chars, self.meta.c_dim))

        # load pretrained dynet model
        if model:
            self.model.populate('%s.dy' %model)

    def word_rep(self, batch):
        batch_embs = [[] for _ in range(len(batch[0][0]))]
        for sample, tag in batch:
            for i,word in enumerate(sample):
                if word not in self.wvm:
                    batch_embs[i].append(self.unk)
                elif self.eval:
                    batch_embs[i].append(self.wvm[word])
                else:
                    batch_embs[i].append(self.wvm[word])
        return [dy.inputTensor(emb) for emb in batch_embs]

    def enable_dropout(self):
        self.fwdRNN.set_dropout(0.3)
        self.bwdRNN.set_dropout(0.3)
        self.fwdRNN2.set_dropout(0.3)
        self.bwdRNN2.set_dropout(0.3)

    def disable_dropout(self):
        self.fwdRNN.disable_dropout()
        self.bwdRNN.disable_dropout()
        self.fwdRNN2.disable_dropout()
        self.bwdRNN2.disable_dropout()

    def initialize_paramerets(self):
        # apply dropout
        if self.eval:
            self.disable_dropout()
        else:
            self.enable_dropout()

        # initialize the RNNs
        self.f_init = self.fwdRNN.initial_state()
        self.b_init = self.bwdRNN.initial_state()
        self.f2_init = self.fwdRNN2.initial_state()
        self.b2_init = self.bwdRNN2.initial_state()

    def build_tagging_graph(self, batch):
        self.initialize_paramerets()
        # get the word vectors.
        batch_embs = self.word_rep(batch)

        # feed word vectors into biLSTM
        fw_exps = self.f_init.transduce(batch_embs)
        bw_exps = self.b_init.transduce(reversed(batch_embs))
    
        # biLSTM states
        bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]

        # feed word vectors into 2nd biLSTM
        fw_exps = self.f2_init.transduce(bi_exps)
        bw_exps = self.b2_init.transduce(reversed(bi_exps))
    
        # biLSTM states
        bi_exps = dy.concatenate([dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))], d=1)
        aT = self.meta.activation(self.aw * bi_exps + self.ab)
        alpha = self.av * aT
        attn = dy.softmax(alpha, 1)
        weighted_sum = dy.reshape(bi_exps * dy.transpose(attn), (self.meta.lstm_word_dim*2, ))
        if not self.eval:
            weighted_sum = dy.dropout(weighted_sum, 0.3)
        xh = self.meta.activation(self.w1 * weighted_sum + self.b1)
        if not self.eval:
            xh = dy.dropout(xh, 0.3)
        xo = self.w2 * xh + self.b2
        return xo

    def sent_loss(self, samples):
        self.eval = False
        vecs = self.build_tagging_graph(samples) # seq_len, ntags, batch_size 
        tids = [self.meta.t2i[tag] for sample,tag in samples]
        self.loss.append(dy.pickneglogsoftmax_batch(vecs, tids))

    def tag_sent(self, samples):
        dy.renew_cg()
        self.eval = True
        probs = self.build_tagging_graph(samples)
        probs = probs.npvalue()
        btags = []
        if probs.ndim == 1:
            probs = probs[:, np.newaxis]
        for i in range(len(probs[0])):
            tag = np.argmax(probs[:, i])
            btags.append(str(self.meta.i2t[tag]))
            #print(' '.join(samples[i][0])+'\t'+samples[i][1]+'\t'+str(tag)+'\t'+str(tag==samples[i][1]))
        return btags

def make_batches(args, samples):
    data = {}
    bdata = []
    for sent in samples:
        bucket = len(sent[0])
        data.setdefault(bucket, [])
        data[bucket].append(sent)
    data = data.values()
    for bucket in data:
        n = max(1, int(len(bucket) / args.batch_size))
        bdata += [bucket[i::int(n)] for i in range(n)]
    return bdata

def read_train_test_dev(args):
    train = []
    dev = []
    test = []
    with io.open(args.rev_norev_file, encoding='utf-8') as fp:
        for i,line in enumerate(fp):
            fset, text, tag = line.strip().split('\t')
            text = ['<'] + text.split() + ['>']
            if fset == 'TEST':
                test.append((text, tag))
            elif fset == 'DEV':
                dev.append((text, tag))
            else:
                train.append((text, tag))
    return make_batches(args, train), make_batches(args, dev), make_batches(args, test)

def eval(tagger, dev, ofp=None):
    good = bad = 0.0
    gall, pall = [], []
    for batch in dev:
        tagged = tagger.tag_sent(batch)
        bgolds = [s[1] for s in batch]
        pall.extend(tagged)
        gall.extend(bgolds)
        for go,gu in zip(bgolds,tagged):
            if go == gu: good += 1
            else: bad += 1
    print(cr(gall, pall, digits=4))
    print(good/(good+bad))
    sys.stdout.flush()
    return f1_score(gall, pall, average='weighted') #good/(good+bad)

def train_tagger(args, tagger, train, dev):
    pr_acc = 0.0
    n_samples = len(train)
    num_tagged, cum_loss = 0, 0
    status_step = int(500/args.batch_size) + 2
    eval_step = int(200000/args.batch_size)
    print(len(train))
    print(len(dev))
    sys.stdout.flush()
    for ITER in range(args.iter):
        random.shuffle(train)
        tagger.eval = False
        tagger.loss = []
        for i,batch in enumerate(train, 1):
            random.shuffle(batch)
            lb0 = [(s,t) for s,t in batch if t=='0']
            lb1 = [(s,t) for s,t in batch if t=='1']
            diff = abs(len(lb1)-len(lb0))
            if len(lb1) > len(lb0):
                batch.extend(lb0[:diff])
            else:
                batch.extend(lb1[:diff])
            dy.renew_cg()
            if i % status_step == 0 or i == n_samples:   # print status
                tagger.trainer.status()
                print(cum_loss / num_tagged)
                sys.stdout.flush()
                cum_loss, num_tagged = 0, 0
            tagger.sent_loss(batch)
            num_tagged += args.batch_size  #NOTE batch size
            batch_loss = dy.sum_batches(dy.esum(tagger.loss))
            cum_loss += batch_loss.scalar_value()
            batch_loss.backward()
            tagger.trainer.update()
            tagger.loss = []
            dy.renew_cg()
        cum_acc = eval(tagger, dev)
        if cum_acc > pr_acc:
            pr_acc = cum_acc
            print('Save Point:: %d' %ITER)
            if args.save_model:
                tagger.model.save('%s.dy' %args.save_model)
        sys.stdout.flush()
        print("epoch %r finished" % ITER)
        sys.stdout.flush()


def set_labels(args, data):
    trainers = {
        'simsgd'   : dy.SimpleSGDTrainer,
        'cysgd'    : dy.CyclicalSGDTrainer,
        'momsgd'   : dy.MomentumSGDTrainer,
        'adam'     : dy.AdamTrainer,
        'adagrad'  : dy.AdagradTrainer,
        'adadelta' : dy.AdadeltaTrainer,
        'amsgrad'  : dy.AmsgradTrainer
        }
    act_fn = {
        'sigmoid' : dy.logistic,
        'tanh'    : dy.tanh,
        'relu'    : dy.rectify,
        }
    meta = Meta()
    meta.trainer = trainers[args.trainer]
    meta.activation = act_fn[args.act_fn]
    tags = set()
    meta.c2i = {'bos':0, 'eos':1, 'unk':2, 'pad':3}
    cid = len(meta.c2i)
    for batch in data:
      for sent, tag in batch:
        tags.add(tag)
        for word in sent:
            for c in word:
                if c not in meta.c2i:
                    meta.c2i[c] = cid
                    cid += 1
    meta.n_chars = len(meta.c2i)
    meta.n_tags = len(tags)
    meta.i2t = dict(enumerate(tags))
    meta.t2i = {t:i for i,t in meta.i2t.items()}
    return meta

def load_data(args):
    wvm = None
    train, dev, test = read_train_test_dev(args)
    meta = set_labels(args, train)
    if args.save_model:
        pickle.dump(meta, open('%s.meta' %args.save_model, 'wb'))
    return train, dev, test, meta

def main():
    parser = ArgumentParser(description="POS Tagger")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--dynet-gpu')
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-devices')
    parser.add_argument('--dynet-autobatch')
    parser.add_argument('--dynet-seed', dest='seed', type=int, default='127')
    parser.add_argument('--rev_norev_file', help='CONLL/TNT Train file')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--pre_word_vec', help='Pretrained word2vec Embeddings')
    parser.add_argument('--bin_vec', type=int, help='1 if binary embedding file else 0')
    parser.add_argument('--elimit', type=int, default=None, help='load only top-n pretrained word vectors (default=all vectors)')
    parser.add_argument('--trainer', default='amsgrad', help='Trainer [momsgd|adam|adadelta|adagrad|amsgrad]')
    parser.add_argument('--activation', dest='act_fn', default='tanh', help='Activation function [tanh|rectify|logistic]')
    parser.add_argument('--iter', type=int, default=100, help='No. of Epochs')
    group.add_argument('--save-model', dest='save_model', help='Specify path to save model')
    group.add_argument('--load-model', dest='load_model', help='Load Pretrained Model')
    parser.add_argument('--output-file', dest='outfile', default='/tmp/out.txt', help='Output File')
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    wvm = gensim.models.KeyedVectors.load_word2vec_format(args.pre_word_vec, binary=args.bin_vec, limit=args.elimit)

    if args.load_model:
        sys.stdout.write('Loading Models ...\n')
        tagger = Tagger(model=args.load_model, wvm=wvm)
        train, dev, test, meta = load_data(args)
        sys.stdout.write('Done!\n')
        eval(tagger, test)
    else:
        # load data
        train, dev, test, meta = load_data(args)
        meta.w_dim_e = wvm.syn0.shape[1]
        # initialize parser
        tagger = Tagger(meta=meta, wvm=wvm)
        train_tagger(args, tagger, train, dev)

if __name__ == '__main__':
    main()
