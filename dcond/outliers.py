import numpy as np
from sklearn.cluster import KMeans

METRICS = ("cosine", "euclidean")


def _check_normalized(X: np.ndarray) -> None:
    norms = np.linalg.norm(X, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(f"cosine metric needs L2-normalized inputs (norms in [{norms.min():.4f}, {norms.max():.4f}])")


def _normalize(C: np.ndarray) -> np.ndarray:
    return C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)


def _spherical_kmeanspp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(X)
    centers = [X[rng.integers(n)]]
    d = 1.0 - X @ centers[0]
    for _ in range(1, k):
        w = np.clip(d, 0, None) ** 2
        probs = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(X[rng.choice(n, p=probs)])
        d = np.minimum(d, 1.0 - X @ centers[-1])
    return np.stack(centers)


def _spherical_kmeans_single(X, k, rng, max_iter, tol):
    C = _spherical_kmeanspp_init(X, k, rng)
    labels = None
    for _ in range(max_iter):
        sims = X @ C.T
        new_labels = sims.argmax(axis=1)
        fit = sims.max(axis=1)
        for c in np.where(np.bincount(new_labels, minlength=k) == 0)[0]:
            worst = int(np.argmin(fit))
            fit[worst] = np.inf
            new_labels[worst] = c
        new_C = np.zeros_like(C)
        np.add.at(new_C, new_labels, X)
        new_C = _normalize(new_C)
        shift = np.max(1.0 - np.sum(new_C * C, axis=1))
        C = new_C
        if labels is not None and np.array_equal(labels, new_labels) and shift < tol:
            break
        labels = new_labels
    labels = (X @ C.T).argmax(axis=1)
    inertia = float(np.sum(1.0 - (X * C[labels]).sum(axis=1)))
    return labels, C, inertia


def spherical_kmeans(X: np.ndarray, k: int, seed: int = 0, n_init: int = 10, max_iter: int = 300, tol: float = 1e-8):
    _check_normalized(X)
    if k == 1:
        C = _normalize(X.mean(axis=0, keepdims=True))
        return np.zeros(len(X), dtype=int), C, float(np.sum(1.0 - X @ C[0]))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        result = _spherical_kmeans_single(X, k, rng, max_iter, tol)
        if best is None or result[2] < best[2]:
            best = result
    return best


def _relabel_by_size(labels: np.ndarray, centroids: np.ndarray):
    sizes = np.bincount(labels, minlength=len(centroids))
    order = sorted(range(len(centroids)), key=lambda c: (-sizes[c], c))
    remap = np.empty(len(order), dtype=int)
    remap[order] = np.arange(len(order))
    return remap[labels], centroids[order]


def fit_clusters(X: np.ndarray, k: int, metric: str, seed: int = 0, n_init: int = 10):
    if metric == "cosine":
        labels, C, _ = spherical_kmeans(X, k, seed=seed, n_init=n_init)
    elif metric == "euclidean":
        if k == 1:
            labels, C = np.zeros(len(X), dtype=int), X.mean(axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
            labels, C = km.labels_, km.cluster_centers_
    else:
        raise ValueError(metric)
    return _relabel_by_size(np.asarray(labels), np.asarray(C, dtype=np.float64))


def class_centroid(X: np.ndarray, metric: str) -> np.ndarray:
    m = X.mean(axis=0)
    return m / np.linalg.norm(m) if metric == "cosine" else m


def distances(X: np.ndarray, centroids: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return np.clip(1.0 - X @ np.asarray(centroids).T, 0.0, 2.0)
    diff = X[:, None, :] - np.asarray(centroids)[None, :, :]
    return np.linalg.norm(diff, axis=2)


def outlier_flags(dist: np.ndarray, groups: np.ndarray, top_pct: float = 5.0, median_mult: float = 2.0):
    dist, groups = np.asarray(dist, dtype=float), np.asarray(groups)
    thr_top = np.empty_like(dist)
    thr_med = np.empty_like(dist)
    for g in np.unique(groups):
        m = groups == g
        thr_top[m] = np.percentile(dist[m], 100.0 - top_pct)
        thr_med[m] = median_mult * np.median(dist[m])
    return dist >= thr_top, dist > thr_med, thr_top, thr_med


def within_group_stats(dist: np.ndarray, groups: np.ndarray):
    pct = np.empty_like(dist, dtype=float)
    z = np.empty_like(dist, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        d = dist[m]
        ranks = d.argsort().argsort()
        pct[m] = 100.0 * (ranks + 1) / len(d)
        sd = d.std()
        z[m] = (d - d.mean()) / sd if sd > 0 else 0.0
    return pct, z
