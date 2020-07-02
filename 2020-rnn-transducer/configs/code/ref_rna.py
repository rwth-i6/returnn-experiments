"""
Python reference implementation of the Recurrent Neural Aligner.
Author: Ivan Sorokin

Based on the papers:

 - "Recurrent Neural Aligner: An Encoder-Decoder Neural Network Model for Sequence to Sequence Mapping"
    Hasim Sak, et al., 2017

 - "Extending Recurrent Neural Aligner for Streaming End-to-End Speech Recognition in Mandarin"
    Linhao Dong, et al., 2018

"""

import numpy as np
from termcolor import colored

NEG_INF = -float("inf")


def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = np.log(sum(np.exp(a - a_max) for a in args))
    return a_max + lsp


def log_softmax(acts, axis):
    """
    Log softmax over the last axis of the 3D array.
    """
    acts = acts - np.max(acts, axis=axis, keepdims=True)
    probs = np.sum(np.exp(acts), axis=axis, keepdims=True)
    log_probs = acts - np.log(probs)
    return log_probs


def forward_pass(log_probs, labels, blank, label_rep=False):

    T, U, _ = log_probs.shape
    S = T-U+2

    alphas = np.zeros((S, U))

    for u in range(1, U):
        alphas[0, u] = alphas[0, u-1] + log_probs[u-1, u-1, labels[u-1]]

    for t in range(1, S):
        alphas[t, 0] = alphas[t-1, 0] + log_probs[t-1, 0, blank]

    for t in range(1, S):
        for u in range(1, U):
            skip = alphas[t-1, u] + log_probs[t+u-1, u, blank]
            emit = alphas[t, u-1] + log_probs[t+u-1, u-1, labels[u-1]]
            alphas[t, u] = logsumexp(emit, skip)
            if label_rep:  # merge_repeated in ctc loss
                # We add the arc which corresponds to a repeated label
                # i.e.
                same = alphas[t-1, u, log_probs[t+u-1, u-1, labels[u-1]]]
                alphas[t, u] = logsumexp(alphas[t, u], same)



    return alphas, alphas[S-1, U-1]


def backward_pass(log_probs, labels, blank):

    T, U, _ = log_probs.shape
    S = T-U+2

    S1 = S-1
    U1 = U-1

    betas = np.zeros((S, U))

    for i in range(1, U):
        u = U1-i
        betas[S1, u] = betas[S1, u+1] + log_probs[T-i, u, labels[u]]

    for i in range(1, S):
        t = S1-i
        betas[t, U1] = betas[t+1, U1] + log_probs[T-i, U1, blank]

    for i in range(1, S):
        t = S1-i
        for j in range(1, U):
            u = U1-j
            skip = betas[t+1, u] + log_probs[T-i-j, u, blank]
            emit = betas[t, u+1] + log_probs[T-i-j, u, labels[u]]
            betas[t, u] = logsumexp(emit, skip)

    return betas, betas[0, 0]


def analytical_gradient(log_probs, alphas, betas, labels, blank):

    T, U, _ = log_probs.shape
    S = T-U+2

    log_like = betas[0, 0]

    grads = np.full(log_probs.shape, NEG_INF)

    for t in range(S-1):
        for u in range(U):
            grads[t+u, u, blank] = alphas[t, u] + betas[t+1, u] + log_probs[t+u, u, blank] - log_like

    for t in range(S):
        for u, l in enumerate(labels):
            grads[t+u, u, l] = alphas[t, u] + betas[t, u+1] + log_probs[t+u, u, l] - log_like

    return -np.exp(grads)


def numerical_gradient(log_probs, labels, neg_loglike, blank):
    epsilon = 1e-5
    T, U, V = log_probs.shape
    grads = np.zeros_like(log_probs)
    for t in range(T):
        for u in range(U):
            for v in range(V):
                log_probs[t, u, v] += epsilon
                alphas, ll_forward = forward_pass(log_probs, labels, blank)
                grads[t, u, v] = (-ll_forward - neg_loglike) / epsilon
                log_probs[t, u, v] -= epsilon
    return grads


def test():

    np.random.seed(0)

    blank = 0
    vocab_size = 4
    input_len = 5
    output_len = 3
    print("T=%d, U=%d, V=%d" % (input_len, output_len+1, vocab_size))
    inputs = np.random.rand(input_len, output_len + 1, vocab_size)
    labels = np.random.randint(1, vocab_size, output_len)

    log_probs = log_softmax(inputs, axis=2)
    print("log-probs:", log_probs.shape)
    print(log_probs[...,0])

    alphas, ll_forward = forward_pass(log_probs, labels, blank)
    print("alphas")
    print(alphas)
    print("LL forward")
    print(ll_forward)
    betas, ll_backward = backward_pass(log_probs, labels, blank)

    assert np.allclose(ll_forward, ll_backward, atol=1e-12, rtol=1e-12), \
        "Log-likelihood from forward and backward pass mismatch."
    print("LL forward == LL backward:    %s" % colored("MATCH", "green"))

    neg_loglike = -ll_forward

    analytical_grads = analytical_gradient(log_probs, alphas, betas, labels, blank)
    numerical_grads = numerical_gradient(log_probs, labels, neg_loglike, blank)

    assert np.allclose(analytical_grads, numerical_grads, atol=1e-6, rtol=1e-6), \
        "Analytical and numerical computation of gradient mismatch."
    print("analytical == numerical grad: %s" % colored("MATCH", "green"))


if __name__ == "__main__":
    test()
