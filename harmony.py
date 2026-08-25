"""
Smolensky Harmony Theory over a PMI lexicon.

Two implementations:

  HarmonicGrammar  -- flat, one-layer.  Closest to what you already wrote, but
                      with the kappa-normalised matching criterion, an explicit
                      singleton (markedness) term, and stochastic annealing.

  Harmonium        -- the actual Harmony Theory architecture: bipartite, with
                      bipolar representation features (lemmas, +/-1) and binary
                      knowledge atoms (collocation patterns, 0/1).

--------------------------------------------------------------------------
Smolensky (1986), ch. 6 of PDP vol. 1
--------------------------------------------------------------------------

    H_K(r, a) = sum_alpha  sigma_alpha * a_alpha * h_kappa(r, k_alpha)

    h_kappa(r, k_alpha) = (r . k_alpha) / |k_alpha|  -  kappa

    |k_alpha| = sum_i |(k_alpha)_i|          (L1 norm, NOT the count of units)
    kappa in [-1, 1]

The L1 normalisation is the part that is missing from nn.py.  It makes the
match score a FRACTION rather than a SUM, so kappa is the fraction of the
constraint that must be satisfied before activating it:

    kappa = -1  ->   0% criterion   (activate everything)
    kappa =  0  ->  50% criterion
    kappa = +1  -> 100% criterion   (activate nothing)

Smolensky flags kappa = 0 as already too permissive.  A raw unnormalised sum
compared against a constant is the kappa -> -1 limit, which is why nn.py
activates the entire lexicon.

Dynamics (Theorem 2, Realizability):

    prob(value = 1) = 1 / (1 + exp(-I/T))

with T annealed to 0.  This is the guarantee you actually want: if T is
lowered slowly enough, the system reaches a GLOBAL harmony maximum with
probability 1.  Deterministic thresholding is the T = 0 quench, which has no
such guarantee -- it halts at the nearest local maximum.

Input features are clamped throughout the computation (Smolensky's completion
task), so the clamping in nn.py is correct and canonical.
"""

import numpy as np
import scipy.sparse as sp


def _build_W(pmi_df, col_x="word_x", col_y="word_y", col_w="pmi",
             mode="raw", shift=0.0):
    x = pmi_df[col_x].to_numpy()
    y = pmi_df[col_y].to_numpy()
    w = pmi_df[col_w].to_numpy(dtype=float)

    vocab = sorted(set(x.tolist()) | set(y.tolist()))
    w2i = {t: i for i, t in enumerate(vocab)}
    N = len(vocab)

    if mode == "ppmi":
        w = np.maximum(w, 0.0)
    elif mode == "sppmi":
        w = np.maximum(w - shift, 0.0)

    i = np.fromiter((w2i[t] for t in x), dtype=np.int64, count=len(x))
    j = np.fromiter((w2i[t] for t in y), dtype=np.int64, count=len(y))
    keep = i != j
    i, j, w = i[keep], j[keep], w[keep]

    lo, hi = np.minimum(i, j), np.maximum(i, j)
    order = np.lexsort((np.abs(w), hi, lo))
    lo, hi, w = lo[order], hi[order], w[order]
    key = lo.astype(np.int64) * N + hi
    last = np.ones(len(key), dtype=bool)
    last[:-1] = key[1:] != key[:-1]
    lo, hi, w = lo[last], hi[last], w[last]

    U = sp.coo_matrix((w, (lo, hi)), shape=(N, N)).tocsr()
    U.eliminate_zeros()
    W = (U + U.T).tocsr()
    W.setdiag(0.0)
    W.eliminate_zeros()
    W.sort_indices()
    return W, vocab, w2i


# ===========================================================================
#  1.  Flat harmonic grammar
# ===========================================================================

class HarmonicGrammar:
    """
    H(s) = 1/2 s' W s  +  b . s        (MAXIMISED)

    b is the singleton / markedness term that nn.py omits.  Two principled
    choices, which can be combined:

      b_i = -kappa * L1_i     Smolensky's matching criterion, L1_i = sum_j |W_ij|.
                              Equivalent to normalising the field by L1_i and
                              thresholding at kappa.

      b_i = log p(w_i)        Required for coherence if the pairwise weights are
                              PMI.  The log-linear (maximum-entropy) expansion of
                              a joint distribution is

                                  log P(w_1..w_n) ~ sum_i log p(w_i)
                                                  + sum_{i<j} PMI(w_i, w_j)

                              PMI is DEFINED as log[p(x,y)/(p(x)p(y))], so the
                              unigram log-probabilities are the first-order part
                              of the same expansion.  Dropping them keeps the
                              interaction term and discards the cost of
                              activating a word at all.  That cost is what
                              stops the lexicon from switching on.
    """

    def __init__(self, pmi_df, unigram_logp=None, mode="raw", shift=0.0, **kw):
        self.W, self.vocab, self.word_to_idx = _build_W(
            pmi_df, mode=mode, shift=shift, **kw)
        self.N = len(self.vocab)
        self.idx_to_word = {i: t for t, i in self.word_to_idx.items()}
        self.L1 = np.asarray(abs(self.W).sum(axis=1)).ravel()
        self.L1[self.L1 == 0] = 1.0

        if unigram_logp is None:
            self.logp = np.zeros(self.N)
        else:
            self.logp = np.array([unigram_logp.get(w, np.log(1e-8))
                                  for w in self.vocab])

    def bias(self, kappa=0.0, beta=0.0, mu=0.0):
        """beta scales the unigram term, mu is a flat activation cost."""
        return -kappa * self.L1 + beta * self.logp - mu

    def harmony(self, s, lam=0.0, **kw):
        """
        H(s) = 1/2 s'Ws + b.s - lam/2 * n^2

        The -lam/2 n^2 term is a *STRUCT / economy-of-representation constraint:
        a markedness constraint penalising the amount of activated structure
        itself.  It is the only term that scales with n the way the interaction
        term does, which is why no per-unit bias can substitute for it.
        """
        n = float(s.sum())
        return (0.5 * float(s @ (self.W @ s)) + float(self.bias(**kw) @ s)
                - 0.5 * lam * n * n)

    def normalized_harmony(self, s, **kw):
        """Per-active-unit harmony; raw H is O(n^2) and not comparable across sizes."""
        n = s.sum()
        return self.harmony(s, **kw) / n if n else 0.0

    def score(self, word_list, n_null=500, rng=None):
        """
        Well-formedness of an OBSERVED word set: mean pairwise harmony, plus a
        z-score against size-matched random sets.

        This needs no dynamics at all.  If what you want is a harmony VALUE for
        attested linguistic material, this is the whole computation -- the
        completion dynamics is only needed to FIND a maximum-harmony state.
        """
        rng = np.random.default_rng(rng)
        idx = [self.word_to_idx[w] for w in word_list if w in self.word_to_idx]
        n = len(idx)
        if n < 2:
            raise ValueError("need at least 2 in-vocabulary words")

        def pair_h(ii):
            s = np.zeros(self.N); s[ii] = 1.0
            return 0.5 * float(s @ (self.W @ s)) / (n * (n - 1) / 2)

        obs = pair_h(idx)
        null = np.array([pair_h(rng.choice(self.N, n, replace=False))
                         for _ in range(n_null)])
        return {"harmony_per_pair": obs,
                "null_mean": float(null.mean()),
                "null_sd": float(null.std()),
                "z": float((obs - null.mean()) / null.std())}

    def anneal(self, seed_words=(), kappa=0.0, beta=0.0, mu=0.0, lam=0.0,
               T0=2.0, T1=0.01, n_sweeps=60, rng=None, verbose=False):
        """Stochastic completion with geometric cooling (Theorem 2)."""
        rng = np.random.default_rng(rng)
        b = self.bias(kappa=kappa, beta=beta, mu=mu)

        s = (rng.random(self.N) < 0.05).astype(float)
        clamped = np.array([self.word_to_idx[w] for w in seed_words
                            if w in self.word_to_idx], dtype=np.int64)
        s[clamped] = 1.0
        free = np.setdiff1d(np.arange(self.N), clamped)

        indptr, indices, data = self.W.indptr, self.W.indices, self.W.data
        schedule = T0 * (T1 / T0) ** (np.arange(n_sweeps) / max(n_sweeps - 1, 1))

        kw = dict(kappa=kappa, beta=beta, mu=mu)
        best_s, best_H = s.copy(), self.harmony(s, lam=lam, **kw)
        n = float(s.sum())
        for T in schedule:
            for i in rng.permutation(free):
                lo, hi = indptr[i], indptr[i + 1]
                I = (float(data[lo:hi] @ s[indices[lo:hi]]) + b[i]
                     - lam * (n - s[i]) - 0.5 * lam)
                p = 1.0 / (1.0 + np.exp(-np.clip(I / T, -60.0, 60.0)))
                new = 1.0 if rng.random() < p else 0.0
                n += new - s[i]
                s[i] = new
            H = self.harmony(s, lam=lam, **kw)
            if H > best_H:
                best_H, best_s = H, s.copy()
        if verbose:
            print(f"  H*={best_H:.2f}  n={int(best_s.sum())}")
        active = np.flatnonzero(best_s == 1.0)
        return [self.idx_to_word[i] for i in active], best_s, best_H


# ===========================================================================
#  2.  The harmonium
# ===========================================================================

class Harmonium:
    """
    Bipartite Harmony Theory network.

      lower layer : representation features r_i in {-1, +1}   -- lemmas
      upper layer : knowledge atoms a_alpha in {0, 1}         -- collocations

    This dissolves the bipolar-vs-binary question that nn.py ran into.  Harmony
    Theory uses BOTH encodings; they are assigned by role, not chosen between.
    Features are bipolar so that mismatch is penalised (a -1 in r against a +1
    in k_alpha subtracts).  Atoms are binary because an atom is either recruited
    into the current schema or it is not.

    Atoms are built here from the PMI graph: atom alpha_i is the top-m PMI
    neighbourhood of word i, with strength sigma_i = mean PMI over that
    neighbourhood.  Strongly negative-PMI words enter the atom with -1, so an
    atom encodes both what should and should not co-occur.
    """

    def __init__(self, pmi_df, m=8, neg_m=2, mode="raw", **kw):
        self.W, self.vocab, self.word_to_idx = _build_W(pmi_df, mode=mode, **kw)
        self.N = len(self.vocab)
        self.idx_to_word = {i: t for t, i in self.word_to_idx.items()}

        rows, cols, vals, sigma = [], [], [], []
        Wc = self.W.tocsr()
        for i in range(self.N):
            lo, hi = Wc.indptr[i], Wc.indptr[i + 1]
            nbr, wt = Wc.indices[lo:hi], Wc.data[lo:hi]
            if len(nbr) < 2:
                continue
            pos = np.argsort(-wt)[:m]
            pos = pos[wt[pos] > 0]
            if len(pos) == 0:
                continue
            neg = np.argsort(wt)[:neg_m]
            neg = neg[wt[neg] < 0]

            a = len(sigma)
            rows.append(a); cols.append(i); vals.append(1.0)
            for p in pos:
                rows.append(a); cols.append(nbr[p]); vals.append(1.0)
            for q in neg:
                rows.append(a); cols.append(nbr[q]); vals.append(-1.0)
            sigma.append(float(wt[pos].mean()))

        self.M = len(sigma)
        self.K = sp.coo_matrix((vals, (rows, cols)),
                               shape=(self.M, self.N)).tocsr()
        self.sigma = np.array(sigma)
        self.Knorm = np.asarray(abs(self.K).sum(axis=1)).ravel()   # |k_alpha|
        # W_{i,alpha} = sigma_alpha (k_alpha)_i / |k_alpha|
        self.Wia = (sp.diags(self.sigma / self.Knorm) @ self.K).T.tocsr()

    def harmony(self, r, a, kappa=0.0):
        match = (self.K @ r) / self.Knorm - kappa
        return float(self.sigma @ (a * match))

    def max_harmony(self, kappa=0.0):
        """Upper bound: every atom perfectly matched. Use to normalise H."""
        return float(self.sigma[self.sigma > 0].sum()) * (1.0 - kappa)

    def complete(self, seed_words, kappa=0.3, T0=1.5, T1=0.02,
                 n_sweeps=80, rng=None):
        """Block Gibbs (atoms in parallel, then features) with annealing."""
        rng = np.random.default_rng(rng)
        r = rng.choice([-1.0, 1.0], size=self.N)
        clamped = np.array([self.word_to_idx[w] for w in seed_words
                            if w in self.word_to_idx], dtype=np.int64)
        if len(clamped) == 0:
            raise ValueError("no seed word in vocabulary")
        r[clamped] = 1.0

        schedule = T0 * (T1 / T0) ** (np.arange(n_sweeps) / max(n_sweeps - 1, 1))
        best = (None, None, -np.inf)
        for T in schedule:
            I_a = self.sigma * ((self.K @ r) / self.Knorm - kappa)
            a = (rng.random(self.M) < 1.0 / (1.0 + np.exp(-I_a / T))).astype(float)

            I_r = 2.0 * (self.Wia @ a)
            r = np.where(rng.random(self.N) < 1.0 / (1.0 + np.exp(-I_r / T)),
                         1.0, -1.0)
            r[clamped] = 1.0

            H = self.harmony(r, a, kappa)
            if H > best[2]:
                best = (r.copy(), a.copy(), H)

        r, a, H = best
        active = [self.idx_to_word[i] for i in np.flatnonzero(r > 0)
                  if i not in set(clamped.tolist())]
        return active, H, H / self.max_harmony(kappa), int(a.sum())


# ===========================================================================

def _synthetic(n_clusters=8, per_cluster=50, p_in=0.35, p_out=0.01, rng=None):
    import pandas as pd
    rng = np.random.default_rng(rng)
    N = n_clusters * per_cluster
    words = [f"c{c}_w{k}" for c in range(n_clusters) for k in range(per_cluster)]
    lab = np.repeat(np.arange(n_clusters), per_cluster)
    rows = []
    for i in range(N):
        for j in range(i + 1, N):
            same = lab[i] == lab[j]
            if rng.random() < (p_in if same else p_out):
                rows.append((words[i], words[j],
                             rng.normal(2.5, .8) if same else rng.normal(-.3, .8)))
    # plausible Zipfian unigram log-probs
    ranks = np.arange(1, N + 1)
    p = (1.0 / ranks) / (1.0 / ranks).sum()
    rng.shuffle(p)
    return (pd.DataFrame(rows, columns=["word_x", "word_y", "pmi"]),
            lab, words, dict(zip(words, np.log(p))))


if __name__ == "__main__":
    df, lab, words, logp = _synthetic(rng=0)
    truth = set(w for w, l in zip(words, lab) if l == 0)
    seeds = [words[0], words[1]]

    hg = HarmonicGrammar(df, unigram_logp=logp)
    print(f"N={hg.N}  edges={hg.W.nnz//2}  "
          f"L1 row norm: median={np.median(hg.L1):.1f} max={hg.L1.max():.1f}\n")

    print("--- flat model: kappa sweep (Smolensky's matching criterion) ---")
    print(f"{'kappa':>6} {'criterion':>10} {'n active':>9} {'prec':>6} {'rec':>6} {'H/n':>8}")
    for kappa in [-0.5, 0.0, 0.10, 0.20, 0.30, 0.40, 0.60]:
        got, s, H = hg.anneal(seeds, kappa=kappa, rng=1)
        got = set(got)
        prec = len(got & truth) / max(len(got), 1)
        rec = len(got & truth) / len(truth)
        print(f"{kappa:6.2f} {(kappa+1)/2:9.0%} {int(s.sum()):9d} "
              f"{prec:6.2f} {rec:6.2f} {hg.normalized_harmony(s, kappa=kappa):8.2f}")

    print("\n--- flat model: unigram term alone (kappa = 0, no free threshold) ---")
    for beta in [0.0, 0.5, 1.0, 2.0]:
        got, s, H = hg.anneal(seeds, kappa=0.0, beta=beta, rng=1)
        got = set(got)
        prec = len(got & truth) / max(len(got), 1)
        print(f"beta={beta:4.1f} -> n={int(s.sum()):4d}  prec={prec:.2f}  "
              f"rec={len(got & truth)/len(truth):.2f}")

    print("\n--- harmonium (bipartite, bipolar features + binary atoms) ---")
    hm = Harmonium(df, m=8, neg_m=2)
    print(f"{hm.M} knowledge atoms over {hm.N} features")
    for kappa in [0.0, 0.2, 0.4, 0.6]:
        got, H, Hn, na = hm.complete(seeds, kappa=kappa, rng=2)
        got = set(got)
        prec = len(got & truth) / max(len(got), 1)
        print(f"kappa={kappa:4.2f} -> n={len(got):4d} atoms={na:4d} "
              f"prec={prec:.2f} rec={len(got & truth)/len(truth):.2f}  "
              f"H={H:8.1f}  H/Hmax={Hn:.3f}")

    print("\n--- harmony as a well-formedness score (kappa=0.4) ---")
    rng = np.random.default_rng(7)
    coh = np.zeros(hg.N); coh[[words.index(w) for w in list(truth)[:12]]] = 1
    inc = np.zeros(hg.N); inc[rng.choice(hg.N, 12, replace=False)] = 1
    print(f"  coherent 12-word set   H/n = {hg.normalized_harmony(coh, kappa=0.4):7.2f}")
    print(f"  random   12-word set   H/n = {hg.normalized_harmony(inc, kappa=0.4):7.2f}")
