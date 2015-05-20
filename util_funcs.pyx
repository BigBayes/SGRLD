from __future__ import division
cimport cython
import numpy as np
cimport numpy as np
ctypedef np.float64_t dtype_t
ctypedef np.uint32_t uitype_t

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_z_ids(np.ndarray[dtype_t, ndim=2] Adk_avg,
                 np.ndarray[dtype_t, ndim=2] Bkw_avg,
                 np.ndarray[uitype_t, ndim=2] Adk,
                 np.ndarray[uitype_t, ndim=2] Bkw,
                 np.ndarray[dtype_t, ndim=2] phi,
                 np.ndarray[dtype_t, ndim=1] uni_rvs,
                 list doc_dicts,
                 list z,
                 double alpha,
                 int num_sim,
                 int burn_in):

    if not phi.flags.f_contiguous: phi = phi.copy('F')
    if not Adk.flags.c_contiguous: phi = phi.copy('C')
    #if not Bkw.flags.f_contiguous: phi = phi.copy('F')

    cdef Py_ssize_t D = Adk.shape[0]
    cdef Py_ssize_t K = Adk.shape[1]
    cdef Py_ssize_t W = Bkw.shape[1]
    cdef Py_ssize_t d, w, i, k, sim, word_cnt, zInit, zOld, zNew
    cdef Py_ssize_t rc_start = 0, rc_mid, rc_stop = K
    cdef double prob_sum, uni_rv
    cdef Py_ssize_t uni_idx = 0
    cdef np.ndarray[dtype_t, ndim=1] probs = np.zeros(K)
    cdef np.ndarray[dtype_t, ndim=1] cumprobs = np.linspace(0,1,K+1)[0:K]
    cdef np.ndarray[uitype_t, ndim=1] zdw

    # Make sure the counts are initialised to zero
    Adk.fill(0)
    Bkw.fill(0)
    Adk_avg.fill(0)
    Bkw_avg.fill(0)
    # Initialise the z_id for each document in the batch
    for d in range(D):
        for w in doc_dicts[d]:
            word_cnt = doc_dicts[d][w]
            zdw = np.zeros(word_cnt,dtype=np.uint32)
            for i in range(word_cnt): #z[d][w]:
                uni_rv = uni_rvs[uni_idx] #np.random.rand() * prob_sum
                uni_idx += 1
                rc_start = 0
                rc_stop  = K
                while rc_start < rc_stop - 1:
                    rc_mid = (rc_start + rc_stop) // 2
                    if cumprobs[rc_mid] <= uni_rv:
                        rc_start = rc_mid
                    else:
                        rc_stop = rc_mid
                #while uni_rv > cumprobs[rc_start]:
                #    rc_start += 1
                zInit    = rc_start
                Adk[d,zInit] += 1
                Bkw[zInit,w] += 1
                zdw[i] = zInit
            z[d][w] = zdw

    # Draw samples from the posterior on z_ids using Gibbs sampling
    for sim in range(num_sim):
        for d in range(0, D):
            for w in doc_dicts[d]:
                word_cnt = doc_dicts[d][w]
                zdw = z[d][w]
                for i in range(word_cnt):
                    zOld = zdw[i]
                    prob_sum = 0
                    # Faster than using numpy elt product
                    for k in range(K):
                        cumprobs[k] = prob_sum
                        prob_sum +=  (alpha + Adk[d,k] - (k == zOld)) * phi[k,w]
                    uni_rv = prob_sum  * uni_rvs[uni_idx]
                    uni_idx += 1

                    # inline randcat function call
                    rc_start = 0
                    rc_stop  = K
                    while rc_start < rc_stop - 1:
                        rc_mid = (rc_start + rc_stop) // 2
                        if cumprobs[rc_mid] <= uni_rv:
                            rc_start = rc_mid
                        else:
                            rc_stop = rc_mid
                    #while uni_rv > cumprobs[rc_start]:
                    #    rc_start += 1

                    zNew = rc_start
                    zdw[i] = zNew
                    Adk[d,zOld]     = Adk[d,zOld] - 1
                    Adk[d,zNew]     = Adk[d,zNew] + 1
                    Bkw[zOld,w]     = Bkw[zOld,w] - 1
                    Bkw[zNew,w]     = Bkw[zNew,w] + 1

                z[d][w] = zdw
        if sim >= burn_in:
            Adk_avg += Adk
            Bkw_avg += Bkw

    Adk_avg /= (num_sim - burn_in)
    Bkw_avg /= (num_sim - burn_in)
