#!/usr/bin/env python3
"""
kmeans_quant.py

K-means clustering based rate-distortion quantization for images.
Usage example:
    python kmeans_quant.py --images path/to/img1.png path/to/img2.jpg --ks 2 4 8 16 32 --outdir out_quant

Dependencies: Pillow, numpy, matplotlib, argparse
(Only standard pip installs; code doesn't require scikit-learn.)
"""
import os
import sys
import argparse
from math import log2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# K-means (Lloyd's algorithm)
# -----------------------------
def kmeans_numpy(X, K, max_iter=100, tol=1e-4, sample_for_init=20000, random_state=None):
    """
    X: (N, D) float array, values expected in [0,255] or normalized; we'll use floats.
    K: number of clusters
    Returns: centroids (K, D), labels (N,)
    """
    rng = np.random.default_rng(random_state)
    N, D = X.shape

    # initialize centroids by sampling random points from X (or sample subset if huge)
    if N > sample_for_init:
        init_idx = rng.choice(N, size=sample_for_init, replace=False)
        init_sample = X[init_idx]
        centroids = init_sample[rng.choice(len(init_sample), size=K, replace=False)].astype(float)
    else:
        centroids = X[rng.choice(N, size=K, replace=False)].astype(float)

    labels = np.zeros(N, dtype=np.int64)
    for it in range(max_iter):
        # assignment step - compute squared distances
        # distances shape (N, K)
        # use broadcasting for efficiency
        diff = X[:, None, :] - centroids[None, :, :]  # (N, K, D)
        dist2 = np.sum(diff * diff, axis=2)  # (N, K)
        new_labels = np.argmin(dist2, axis=1)

        # update step
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(K, dtype=np.int64)
        for k in range(K):
            members = X[new_labels == k]
            if len(members) > 0:
                new_centroids[k] = members.mean(axis=0)
                counts[k] = len(members)
            else:
                # empty cluster: reinitialize to a random data point
                new_centroids[k] = X[rng.integers(0, N)]

        # check centroid shift
        shift = np.sqrt(np.sum((centroids - new_centroids) ** 2, axis=1)).max()
        centroids = new_centroids
        labels = new_labels
        if shift <= tol:
            break

    return centroids, labels

# -----------------------------
# Image helpers
# -----------------------------
def load_image_as_array(path):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype(np.float32)  # shape (H,W,3)
    return arr

def save_array_as_image(arr, path):
    arr_clipped = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr_clipped, mode='RGB')
    img.save(path)

def quantize_image_with_kmeans(img_array, K, **kmeans_kwargs):
    H, W, C = img_array.shape
    X = img_array.reshape(-1, C)  # (N,3)
    centroids, labels = kmeans_numpy(X, K, **kmeans_kwargs)
    Xq = centroids[labels]
    quantized = Xq.reshape(H, W, C)
    return quantized, centroids, labels

# -----------------------------
# Metrics
# -----------------------------
def mse(orig, recon):
    """Mean Squared Error over all channels (per pixel average)."""
    err = (orig.astype(np.float64) - recon.astype(np.float64)) ** 2
    return float(err.mean())

def psnr_from_mse(mse_val, peak=255.0):
    if mse_val == 0:
        return float('inf')
    return 10.0 * np.log10((peak * peak) / mse_val)

# -----------------------------
# Main processing pipeline
# -----------------------------
def process_image(path, ks, outdir, kmeans_kwargs):
    arr = load_image_as_array(path)
    basename = os.path.splitext(os.path.basename(path))[0]
    results = []

    for K in ks:
        print(f"Quantizing '{basename}' with K={K} ...")
        quantized, centroids, labels = quantize_image_with_kmeans(arr, K, **kmeans_kwargs)

        # compute metrics
        D = mse(arr, quantized)
        psnr = psnr_from_mse(D)
        rate = log2(K)  # bits per pixel approximation (naive)
        results.append({'K': K, 'rate_bpp': rate, 'mse': D, 'psnr_db': psnr})

        # save quantized image
        outpath = os.path.join(outdir, f"{basename}_K{K}.png")
        save_array_as_image(quantized, outpath)
        print(f"  saved quantized image -> {outpath}")
        print(f"  MSE={D:.3f}, PSNR={psnr:.2f} dB, rate≈{rate:.2f} bits/pixel")

    return results

def plot_rd_curve(results, title, outpath=None):
    # results: list of dicts with 'rate_bpp' and 'mse'
    rates = [r['rate_bpp'] for r in results]
    dists = [r['mse'] for r in results]

    plt.figure(figsize=(6,4))
    plt.plot(rates, dists, marker='o')
    plt.xlabel('Rate (bits per pixel)')
    plt.ylabel('Distortion (MSE)')
    plt.title(title)
    plt.grid(True)
    if outpath:
        plt.savefig(outpath, bbox_inches='tight')
        print(f"Saved RD plot -> {outpath}")
    plt.show()

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="K-means image quantization (rate-distortion curve).")
    p.add_argument('--images', nargs='+', required=True, help='Paths to input images.')
    p.add_argument('--ks', nargs='+', type=int, required=True, help='List of K values, e.g. --ks 2 4 8 16 32')
    p.add_argument('--outdir', default='quantized_out', help='Output directory to save quantized images and plots.')
    p.add_argument('--max-iter', type=int, default=60, help='Max iterations for k-means.')
    p.add_argument('--tol', type=float, default=1e-3, help='Convergence tolerance.')
    p.add_argument('--sample-for-init', type=int, default=20000, help='Subsample size to build initial centroids when image huge.')
    p.add_argument('--seed', type=int, default=0, help='Random seed.')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    ks_sorted = sorted(set(args.ks))
    kmeans_kwargs = {'max_iter': args.max_iter, 'tol': args.tol, 'sample_for_init': args.sample_for_init, 'random_state': args.seed}

    for img_path in args.images:
        if not os.path.isfile(img_path):
            print(f"Warning: file not found: {img_path} -- skipping.")
            continue
        results = process_image(img_path, ks_sorted, args.outdir, kmeans_kwargs)
        # Save a simple CSV summary
        summary_csv = os.path.join(args.outdir, os.path.splitext(os.path.basename(img_path))[0] + '_rd.csv')
        with open(summary_csv, 'w') as f:
            f.write('K,rate_bpp,mse,psnr_db\n')
            for r in results:
                f.write(f"{r['K']},{r['rate_bpp']:.6f},{r['mse']:.6f},{r['psnr_db']:.6f}\n")
        print(f"Saved CSV summary -> {summary_csv}")

        # Plot RD curve
        rd_plot_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(img_path))[0] + '_rd.png')
        plot_rd_curve(results, title=os.path.basename(img_path) + ' RD curve', outpath=rd_plot_path)

if __name__ == '__main__':
    main()
