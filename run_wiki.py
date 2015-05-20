#!/Users/sam/anaconda/bin/python
from __future__ import division

import argparse
import itertools
import numpy as np
import os

import processwiki as pw
import samplers

home = os.path.expanduser('~')
np.set_printoptions(precision=3, suppress=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('alpha', type=float, default=0.01)
    parser.add_argument('beta', type=float, default=0.0001)
    parser.add_argument('K', type=int, default=50)
    parser.add_argument('batch_size', type=int, default=50)
    parser.add_argument('epsilon', type=float, default=0.01)
    parser.add_argument('tau0', type=float, default=1000)
    parser.add_argument('kappa', type=float, default=0.55)
    parser.add_argument('samples_per_update', type=int, default=50)
    parser.add_argument('num_updates', type=int, default=1000)
    parser.add_argument('output_dir', type=str,default='.')
    args = parser.parse_args()

    step_size_params = (args.epsilon, args.tau0, args.kappa)
    D = 3.3e6 # Approx number of documents in wikipedia

    vocab = pw.create_vocab('wiki.vocab')
    W = len(vocab)

    # Data is a list of form [(article name, article word cts)]
    data = pw.parse_docs(pw.online_wiki_docs(), vocab)
    # Can save data to a file like this
    #pw.dump_list(itertools.islice(data, 1000), 'stored.pkl')
    # And load like this
    #data = pw.load_list('stored.pkl')

    # Use first 1000 docs as held out test set
    test_data = itertools.islice(data, 1000)
    (test_names, holdout_train_cts, holdout_test_cts) = zip(*list(test_data))
    holdout_train_cts = list(holdout_train_cts)
    holdout_train_cts = list(holdout_train_cts)

    batched_cts = pw.take_every(args.batch_size, data)

    print "Running LD Sampler, epsilon = %g, samples per update = %d" % (args.epsilon,
            args.samples_per_update)
    theta0 = np.random.gamma(1,1,(args.K,W))
    ld_func = samplers.LDSampler(D, args.K, W, args.alpha, args.beta, theta0,
            step_size_params, args.output_dir).run_online
    ld_args = (args.num_updates, args.samples_per_update, batched_cts,
            holdout_train_cts, holdout_test_cts)
    ld_func(*ld_args)


if __name__ == '__main__':
    main()
