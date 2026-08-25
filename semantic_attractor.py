"""
Semantic attractor network on a PMI graph.

Fixes the three coupled problems in the original implementation:

  1. The bipolar -> binary switch silently deleted the inhibitory baseline term.
     Bipolar and binary dynamics are NOT different models; they are the same
     model at two different sparsity assumptions:

         sigma = 2s - 1
         h_sigma = W (2s - 1) = 2 W s - R,      R_i = sum_j W_ij  (row sum)

     so "bipolar with threshold 0" == "binary with per-node threshold R_i / 2".
     Dropping to a global threshold of 0 set that baseline to 0, i.e. it moved
     the assumed activity level from a = 0.5 to a = 0.  Hence the swing from
     "everything off" to "everything on".  The general form (Tsodyks &
     Feigel'man 1988) is

         h_i = W_i . (s - a * 1) = (W s)_i - a * R_i

     with a = expected fraction of active units.  For semantic association
     a ~ 0.005-0.05, not 0.5 and not 0.

  2. A single global scalar threshold on a heterogeneous-degree graph gives
     bootstrap-percolation dynamics: a sharp all-or-nothing transition with no
     usable intermediate regime.  Per-node thresholds (a * R_i) plus a global
     inhibition term lam * n make the activity self-regulating.

  3. Comparing a SUM of edge weights against the MEDIAN of a SINGLE edge weight
     is dimensionally inconsistent.  a * R_i has the right units.

Energy (Lyapunov) function actually being minimised:

    E(s) = -0.5 s' W s  +  a * sum_i R_i s_i  +  theta0 * n  +  0.5 * lam * n^2
    with n = sum_i s_i, W symmetric, diag(W) = 0.

Flipping unit i (s_i: 0 -> 1) changes E by

    dE = -h_i + a*R_i + theta0 + lam*n_{-i} + lam/2

so the greedy asynchronous rule "set s_i = 1 iff dE < 0" is exact coordinate
descent on E.  E takes finitely many values on {0,1}^N, so asynchronous updates
converge in finite time.  Clamping seed units preserves this (it is conditional
minimisation over a subspace).
"""

import numpy as np
import scipy.sparse as sp


class SemanticAttractorNetwork:

    def __init__(self, pmi_df, col_x="word_x", col_y="word_y", col_w="pmi",
                 mode="ppmi", shift=0.0, symmetrise="max"):
        """
        mode : 'raw'   keep PMI as-is (negative values included)
               'ppmi'  max(PMI, 0)              -- standard, negative PMI is noise
               'sppmi' max(PMI - shift, 0)      -- shifted PPMI (Levy & Goldberg 2014);
                                                   shift = log(k) for SGNS with k negatives
        symmetrise : how to combine (x,y) and (y,x) duplicates: 'max' | 'mean' | 'first'
        """
        x = pmi_df[col_x].to_numpy()
        y = pmi_df[col_y].to_numpy()
        w = pmi_df[col_w].to_numpy(dtype=float)

        self.vocab = sorted(set(x.tolist()) | set(y.tolist()))
        self.N = len(self.vocab)
        self.word_to_idx = {t: i for i, t in enumerate(self.vocab)}
        self.idx_to_word = {i: t for i, t in enumerate(self.vocab)}

        if mode == "ppmi":
            w = np.maximum(w, 0.0)
        elif mode == "sppmi":
            w = np.maximum(w - shift, 0.0)
        elif mode != "raw":
            raise ValueError(f"unknown mode {mode!r}")

        i = np.fromiter((self.word_to_idx[t] for t in x), dtype=np.int64, count=len(x))
        j = np.fromiter((self.word_to_idx[t] for t in y), dtype=np.int64, count=len(y))

        keep = i != j                       # kill self-loops before they get summed
        i, j, w = i[keep], j[keep], w[keep]

        # Upper-triangular canonical form so (x,y) and (y,x) collide.
        lo = np.minimum(i, j)
        hi = np.maximum(i, j)

        # COO -> CSR sums duplicates. For 'max'/'mean' we deduplicate explicitly.
        if symmetrise == "mean":
            cnt = sp.coo_matrix((np.ones_like(w), (lo, hi)), shape=(self.N, self.N)).tocsr()
            tot = sp.coo_matrix((w, (lo, hi)), shape=(self.N, self.N)).tocsr()
            tot.data /= np.maximum(cnt.data, 1.0)
            U = tot
        elif symmetrise in ("max", "first"):
            order = np.lexsort((w, hi, lo)) if symmetrise == "max" else np.lexsort((hi, lo))
            lo, hi, w = lo[order], hi[order], w[order]
            key = lo.astype(np.int64) * self.N + hi
            last = np.ones(len(key), dtype=bool)
            last[:-1] = key[1:] != key[:-1]      # keep last of each group == max after sort
            lo, hi, w = lo[last], hi[last], w[last]
            U = sp.coo_matrix((w, (lo, hi)), shape=(self.N, self.N)).tocsr()
        else:
            raise ValueError(f"unknown symmetrise {symmetrise!r}")

        U.eliminate_zeros()
        self.W = (U + U.T).tocsr()
        self.W.setdiag(0.0)
        self.W.eliminate_zeros()
        self.W.sort_indices()

        self.R = np.asarray(self.W.sum(axis=1)).ravel()      # signed row sums
        self.deg = np.diff(self.W.indptr).astype(float)      # number of neighbours

    # ---------------------------------------------------------------- energy

    def energy(self, s, a=0.02, lam=0.0, theta0=0.0):
        n = float(s.sum())
        return (-0.5 * float(s @ (self.W @ s))
                + a * float(self.R @ s)
                + theta0 * n
                + 0.5 * lam * n * n)

    # -------------------------------------------------------------- dynamics

    def retrieve(self, seed_words, a=0.02, lam=0.0, theta0=0.0,
                 max_sweeps=100, rng=None, verbose=True, check_energy=True):
        """
        Asynchronous binary updates with clamped seeds.

        a      : assumed activity level -> per-node baseline a * R_i.
                 a = 0.5 reproduces the classical bipolar network exactly.
                 a = 0   reproduces the runaway version.
        lam    : global inhibition strength (soft k-winner-take-all).
                 Fixed points sit near n* where marginal excitation == lam * n.
        theta0 : uniform bias.
        """
        rng = np.random.default_rng(rng)

        s = np.zeros(self.N)
        clamped = np.array(
            [self.word_to_idx[w] for w in seed_words if w in self.word_to_idx],
            dtype=np.int64,
        )
        if len(clamped) == 0:
            raise ValueError("no seed word is in the vocabulary")
        s[clamped] = 1.0
        is_clamped = np.zeros(self.N, dtype=bool)
        is_clamped[clamped] = True
        free = np.flatnonzero(~is_clamped)

        indptr, indices, data = self.W.indptr, self.W.indices, self.W.data
        n = float(s.sum())
        trace = [self.energy(s, a, lam, theta0)]

        for sweep in range(max_sweeps):
            changed = 0
            for i in rng.permutation(free):
                lo, hi = indptr[i], indptr[i + 1]
                h = float(data[lo:hi] @ s[indices[lo:hi]])
                n_minus = n - s[i]
                margin = h - a * self.R[i] - theta0 - lam * n_minus - 0.5 * lam
                new = 1.0 if margin > 0 else (0.0 if margin < 0 else s[i])
                if new != s[i]:
                    n += new - s[i]
                    s[i] = new
                    changed += 1
            trace.append(self.energy(s, a, lam, theta0))
            if check_energy and trace[-1] > trace[-2] + 1e-8:
                raise AssertionError(
                    f"energy increased at sweep {sweep}: "
                    f"{trace[-2]:.6f} -> {trace[-1]:.6f} (W not symmetric?)"
                )
            if changed == 0:
                if verbose:
                    print(f"converged in {sweep + 1} sweeps, "
                          f"n={int(n)}/{self.N} ({n / self.N:.1%}), E={trace[-1]:.3f}")
                break
        else:
            if verbose:
                print(f"no fixed point after {max_sweeps} sweeps, n={int(n)}")

        active = np.flatnonzero(s == 1.0)
        return [self.idx_to_word[i] for i in active], s, np.array(trace)

    # ------------------------------------------------------------ diagnostics

    def sweep_a(self, seed_words, a_grid, lam=0.0, **kw):
        """Activity vs. assumed sparsity. Expect a sharp percolation step at lam=0."""
        out = []
        for a in a_grid:
            words, s, _ = self.retrieve(seed_words, a=a, lam=lam,
                                        verbose=False, **kw)
            out.append((a, int(s.sum())))
        return out

    def seed_specificity(self, seed_sets, **kw):
        """
        The critical test. If retrieved sets barely differ across unrelated
        seeds, the network carries no seed information regardless of how
        plausible the words look.
        """
        sets = []
        for sw in seed_sets:
            words, _, _ = self.retrieve(sw, verbose=False, **kw)
            sets.append(set(words))
        m = len(sets)
        J = np.eye(m)
        for p in range(m):
            for q in range(p + 1, m):
                u = len(sets[p] | sets[q])
                J[p, q] = J[q, p] = (len(sets[p] & sets[q]) / u) if u else 0.0
        return J, sets

    # -------------------------------------------------- graded alternative

    def random_walk_restart(self, seed_words, alpha=0.15, tol=1e-10, max_iter=1000):
        """
        Personalised PageRank on the (P)PMI graph.

            p <- (1 - alpha) * P p + alpha * e_seed,   P = W D^-1, W >= 0

        A contraction with modulus (1 - alpha): unique fixed point, geometric
        convergence, no threshold, no phase transition, output is a RANKED list.
        For "what does this word evoke" this is almost certainly what you want
        instead of a binary attractor.  Requires non-negative weights.
        """
        if self.W.min() < 0:
            raise ValueError("random_walk_restart needs non-negative weights "
                             "(build with mode='ppmi' or 'sppmi')")
        d = np.asarray(self.W.sum(axis=0)).ravel()
        d[d == 0] = 1.0
        P = self.W @ sp.diags(1.0 / d)

        e = np.zeros(self.N)
        idx = [self.word_to_idx[w] for w in seed_words if w in self.word_to_idx]
        if not idx:
            raise ValueError("no seed word is in the vocabulary")
        e[idx] = 1.0 / len(idx)

        p = e.copy()
        for _ in range(max_iter):
            p_new = (1 - alpha) * (P @ p) + alpha * e
            if np.abs(p_new - p).sum() < tol:
                p = p_new
                break
            p = p_new
        order = np.argsort(-p)
        return [(self.idx_to_word[i], float(p[i])) for i in order]


# ------------------------------------------------------------------ self-test

def _synthetic_pmi(n_clusters=8, per_cluster=50, p_in=0.35, p_out=0.01, rng=None):
    """Planted-community PMI graph: strong within-topic, weak across-topic."""
    import pandas as pd
    rng = np.random.default_rng(rng)
    N = n_clusters * per_cluster
    words = [f"c{c}_w{k}" for c in range(n_clusters) for k in range(per_cluster)]
    label = np.repeat(np.arange(n_clusters), per_cluster)
    rows = []
    for i in range(N):
        for j in range(i + 1, N):
            same = label[i] == label[j]
            if rng.random() < (p_in if same else p_out):
                val = rng.normal(2.5, 0.8) if same else rng.normal(0.4, 0.8)
                rows.append((words[i], words[j], val))
    return pd.DataFrame(rows, columns=["word_x", "word_y", "pmi"]), label, words


if __name__ == "__main__":
    df, label, words = _synthetic_pmi(rng=0)
    print(f"synthetic graph: {len(df)} edges")

    net = SemanticAttractorNetwork(df, mode="ppmi")
    print(f"N={net.N}  density={net.W.nnz / net.N**2:.4f}  "
          f"row-sum R: median={np.median(net.R):.2f} max={net.R.max():.2f}")

    seeds = [words[0], words[1]]
    truth = set(w for w, l in zip(words, label) if l == 0)

    print("\n--- reproducing the two failure modes ---")
    for a, tag in [(0.5, "a=0.5  (== classical bipolar)"),
                   (0.0, "a=0.0  (== original binary code)")]:
        _, s, _ = net.retrieve(seeds, a=a, verbose=False)
        print(f"{tag:34s} -> {int(s.sum()):4d}/{net.N} active")

    print("\n--- sparsity-corrected threshold ---")
    for a in [0.02, 0.05, 0.10, 0.20, 0.35]:
        got, s, _ = net.retrieve(seeds, a=a, verbose=False)
        got = set(got)
        prec = len(got & truth) / max(len(got), 1)
        rec = len(got & truth) / len(truth)
        print(f"a={a:4.2f} -> n={int(s.sum()):4d}  precision={prec:.2f}  recall={rec:.2f}")

    print("\n--- with global inhibition (lam) ---")
    for lam in [0.0, 0.05, 0.2, 0.5]:
        got, s, tr = net.retrieve(seeds, a=0.05, lam=lam, verbose=False)
        got = set(got)
        prec = len(got & truth) / max(len(got), 1)
        print(f"lam={lam:4.2f} -> n={int(s.sum()):4d}  precision={prec:.2f}  "
              f"E: {tr[0]:.1f} -> {tr[-1]:.1f}  monotone={bool(np.all(np.diff(tr) <= 1e-8))}")

    print("\n--- seed specificity (Jaccard between retrieved sets) ---")
    seed_sets = [[words[c * 50], words[c * 50 + 1]] for c in range(4)]
    for a, lam, tag in [(0.00, 0.0, "original code"),
                        (0.05, 0.0, "corrected a, no inhibition"),
                        (0.05, 0.2, "corrected a + inhibition")]:
        J, _ = net.seed_specificity(seed_sets, a=a, lam=lam)
        off = J[np.triu_indices(len(seed_sets), 1)]
        print(f"a={a:4.2f} lam={lam:4.2f} ({tag:26s}) -> mean Jaccard = {off.mean():.3f} "
              f"({'NO seed information' if off.mean() > 0.5 else 'seed-specific'})")

    print("\n--- random walk with restart (top 10) ---")
    rank = net.random_walk_restart(seeds, alpha=0.2)
    hits = sum(1 for w, _ in rank[:20] if w in truth)
    for w, p in rank[:10]:
        print(f"  {w:12s} {p:.5f}  {'*' if w in truth else ''}")
    print(f"  top-20 in planted cluster: {hits}/20")
