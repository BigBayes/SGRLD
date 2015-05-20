from __future__ import division

import errno
import numpy as np
import os
import time
import util_funcs

class LDSampler(object):
    """SGRLD using the expanded-mean parameteristation"""
    def __init__(self, D, K, W, alpha, beta, theta0,
            step_size_params, results_dir):
        self.D = D
        self.K = K
        self.W = W
        self.alpha = alpha
        self.beta = beta
        self.step_size_params = step_size_params
        self.update_ct = 0
        self.theta = theta0
        self.phi = theta0 / np.sum(theta0,1)[:,np.newaxis]
        self.results_dir = results_dir

    def sample_counts(self, train_cts, batch_D, num_samples):
        batch_N = sum(sum(ddict.values()) for ddict in train_cts)
        uni_rvs = np.random.uniform(size = (batch_N)*(num_samples+1))
        z  = [{} for d in range(0, batch_D)]
        Adk = np.zeros((batch_D, self.K), dtype = np.uint32)
        Bkw = np.zeros((self.K, self.W), dtype = np.uint32)
        Adk_mean = np.zeros(Adk.shape)
        Bkw_mean = np.zeros(Bkw.shape)
        burn_in = num_samples // 2
        util_funcs.sample_z_ids(Adk_mean, Bkw_mean, Adk, Bkw, self.phi,
                uni_rvs, train_cts, z, self.alpha, num_samples, burn_in)
        return (Adk_mean, Bkw_mean)

    def update(self, train_cts, num_samples):
        batch_D = len(train_cts)
        Adk_mean, Bkw_mean = self.sample_counts(train_cts, batch_D, num_samples)
        phi_star = self.phi.copy()
        theta_star = self.theta.copy()
        (a,b,c) = self.step_size_params
        eps_t  = a*((1 + self.update_ct/b)**-c)
        for k in range(self.K):
            phi_k = self.phi[k,:]
            theta_k = self.theta[k,:]
            z = np.random.randn(self.W)
            # Update theta according to Equation 11 in paper;
            grad = self.beta - theta_k + (self.D/batch_D)*(Bkw_mean[k,:] - np.sum(Bkw_mean[k,:])*phi_k)
            theta_k = np.abs(theta_k + eps_t*grad + (2*eps_t)**.5*z*theta_k**.5)
            theta_star[k,:] = theta_k
            phi_star[k,:] = theta_k / np.sum(theta_k)
        self.phi = phi_star
        self.theta = theta_star
        self.update_ct += 1
        return (Adk_mean, Bkw_mean)

    def run_online(self, num_updates, samples_per_update, batched_cts,
            holdout_train_cts, holdout_test_cts):
        self.log_preds = []
        t = 1
        self.ho_log_preds = []
        self.avg_probs = {(d, w): 0.0 for (d, ctr) in enumerate(holdout_test_cts) for w in ctr}
        self.ho_count = 0
        self.create_output_dir()
        (a,b,c) = self.step_size_params
        params_dict = {'a':a,'b':b,'c':c,'alpha':self.alpha,'beta':self.beta,
                'K':self.K,'samples':samples_per_update,'func':'online'}
        sampler_name = self.__class__.__name__
        for batch in batched_cts:
            if t == num_updates:
                break
            params_dict['batch_size'] = len(batch)
            self.basename = str(params_dict)
            (names, train_cts, test_cts) = zip(*batch)
            train_cts = list(train_cts)
            test_cts = list(test_cts)
            start = time.time()
            (Adk_mean, Bkw_mean) = self.update(train_cts, samples_per_update)
            self.log_preds.append(self.log_pred(test_cts, Adk_mean,Bkw_mean))
            end = time.time()
            docs_so_far = t * len(batch)
            print sampler_name + " iteration %d, docs so far %d, log pred %g, time %g" % (t, docs_so_far, self.log_preds[t-1], end - start)
            # Assess holdout log-pred every 5000 documents
            if docs_so_far % 5000  == 0:
                ho_lp = self.holdout_log_pred(holdout_train_cts, holdout_test_cts, samples_per_update)
                self.ho_log_preds.append([docs_so_far, ho_lp])
                self.save_variables(['ho_log_preds'])
            t += 1

    def log_pred(self, test_cts, Adk_mean, Bkw_mean):
        eta_hat = Adk_mean + self.alpha
        eta_hat /= np.sum(eta_hat, 1)[:, np.newaxis]
        phi_hat = self.phi
        log_probs = {(d, w): cntr[w]*np.log(np.dot(eta_hat[d, :],
                     phi_hat[:, w])) for (d, cntr) in enumerate(test_cts)
                     for w in cntr}
        num_words = sum(sum(cntr.values()) for cntr in test_cts)
        return sum(log_probs.values()) / num_words

    def holdout_log_pred(self, holdout_train_cts, holdout_test_cts, num_samps):
        batch_D = len(holdout_train_cts)
        Adk_mean, Bkw_mean = self.sample_counts(holdout_train_cts, batch_D,
                                                num_samps)
        eta_hat = Adk_mean + self.alpha
        eta_hat /= np.sum(eta_hat, 1)[:, np.newaxis]
        phi_hat = self.phi
        T = self.ho_count
        old_avg = self.avg_probs
        avg_probs = {(d, w): (T*old_avg[(d, w)] +
                              np.dot(eta_hat[d, :], phi_hat[:, w])) / (T+1)
                     for (d, w) in old_avg}
        self.avg_probs = avg_probs
        self.ho_count += 1
        log_avg_probs = {(d, w): cntr[w] * np.log(avg_probs[(d, w)])
                         for (d, cntr) in enumerate(holdout_test_cts) for w in cntr}
        num_words = sum(sum(cntr.values()) for cntr in holdout_test_cts)
        return sum(log_avg_probs.values()) / num_words

    def create_output_dir(self):
        self.dirname = os.path.join(self.results_dir, self.__class__.__name__)
        try:
            os.makedirs(self.dirname)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
            else:
                raise

    def save_variables(self, attrs):
        for attr in attrs:
            np.savetxt(os.path.join(self.dirname,
                                    self.basename+attr+'_LD.dat'),
                       np.array(getattr(self, attr)))

    def store_phi(self, iter_num):
        try:
            self.stored_phis[str(iter_num)] = self.phi
        except AttributeError:
            self.stored_phis = {str(iter_num): self.phi}

    def save_stored_phis(self):
        np.savez(os.path.join(self.dirname, self.basename+'_stored_phis.npz'),
                 **self.stored_phis)
