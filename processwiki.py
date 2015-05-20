# functions to convert wikipedia articles saved as
# (article_as_string, title)
# to
# (title, ({wordid : trainwordct}, {wordid: testwordct}))
# and save them back to disk

from __future__ import division

from collections import Counter, deque
from itertools import islice
from wikirandom import get_random_wikipedia_article

import cPickle as pickle
import re

def online_wiki_docs():
    # This is slow, for illustration only
    while True:
        yield get_random_wikipedia_article()

def load_list(pkl_file, num_items=3300000):
    with open(pkl_file,'rb') as f:
        while num_items > 0:
            try:
                yield pickle.load(f)
                num_items -= 1
            except EOFError:
                raise StopIteration

def clean_doc(doc, vocab):
    doc = doc.lower()
    doc = re.sub(r'-', ' ', doc)
    doc = re.sub(r'[^a-z ]', '', doc)
    doc = re.sub(r' +', ' ', doc)
    return [word for word in doc.split() if word in vocab]

def parse_docs(name_docs,vocab):
    for doc, name in name_docs:
        words = clean_doc(doc, vocab)
        # Hold back every 10th word to get an online estimate of perplexity
        train_words = [vocab[w] for (i,w) in enumerate(words) if i % 10 != 0]
        test_words = [vocab[w] for (i,w) in enumerate(words) if i % 10 == 0]
        train_cntr = Counter(train_words)
        test_cntr = Counter(test_words)
        yield (name, train_cntr, test_cntr)

def take_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))

def group_up(n, N, iterable):
    i = iter(iterable)
    ctr = 0
    while ctr < N:
        piece = list(islice(i,n))
        ctr += n
        yield piece

def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

def save_data(source, out_file):
    with open(out_file,'wb') as out:
        for item in source:
            pickle.dump(item, out, protocol=-1)

def create_vocab(vocab_file):
    dictnostops = open(vocab_file).readlines()
    vocab = dict()
    for idx,word in enumerate(dictnostops):
        word = word.lower()
        word = re.sub(r'[^a-z]', '', word)
        vocab[word] = idx
    return vocab
