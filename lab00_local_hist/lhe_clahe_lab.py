"""LHE 与 CLAHE 的纯 NumPy 实现（不调用两种算法的库函数）。"""
from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
import numpy as np
from PIL import Image
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
def local_hist_equalization(img, window_size=15):
    """逐像素局部直方图均衡化。边界采用反射填充。"""
    if window_size % 2 == 0:
        raise ValueError("window_size 必须是奇数")
    r = window_size // 2
    padded = np.pad(img, r, mode="reflect")
    out = np.empty_like(img)
    area = window_size * window_size
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i + window_size, j:j + window_size]
            hist = np.bincount(window.ravel(), minlength=256)
            cdf = np.cumsum(hist)
            out[i, j] = np.clip(np.rint(255 * cdf[img[i, j]] / area), 0, 255)
    return out
def _clip_histogram(hist, clip_limit):
    """截断直方图，并把超出部分近似均匀地重分配到 256 个灰度级。"""
    hist = hist.astype(np.int64, copy=True)
    excess = int(np.maximum(hist - clip_limit, 0).sum())
    hist = np.minimum(hist, clip_limit)
    q, rem = divmod(excess, 256)
    hist += q
    if rem:
        # 让余数尽量均匀散布，避免只堆到低灰度端。
        idx = np.floor(np.arange(rem) * 256 / rem).astype(int)
        hist[idx] += 1
    return hist
def clahe(img, tile_grid=(8, 8), clip_factor=2.0):
    """手工实现 CLAHE：分块、限幅、CDF 映射、双线性插值。"""
    h, w = img.shape
    nr, nc = tile_grid
    row_edges = np.linspace(0, h, nr + 1, dtype=int)
    col_edges = np.linspace(0, w, nc + 1, dtype=int)
    luts = np.zeros((nr, nc, 256), dtype=np.uint8)
    for r in range(nr):
        for c in range(nc):
            tile = img[row_edges[r]:row_edges[r + 1], col_edges[c]:col_edges[c + 1]]
            hist = np.bincount(tile.ravel(), minlength=256)
            # clip_factor=1 表示平均箱高；至少为 1，保证实现可用。
            limit = max(1, int(np.floor(clip_factor * tile.size / 256)))
            hist = _clip_histogram(hist, limit)
            cdf = np.cumsum(hist)
            luts[r, c] = np.clip(np.rint(255 * cdf / cdf[-1]), 0, 255)
    # 各分块中心坐标；像素在相邻中心之间作双线性插值。
    row_centers = (row_edges[:-1] + row_edges[1:] - 1) / 2
    col_centers = (col_edges[:-1] + col_edges[1:] - 1) / 2
    out = np.empty_like(img)
    cb = np.searchsorted(col_centers, np.arange(w), side="right")
    c0_all = np.maximum(0, cb - 1)
    c1_all = np.minimum(nc - 1, cb)
    den_x = col_centers[c1_all] - col_centers[c0_all]
    wx_all = np.divide(np.arange(w) - col_centers[c0_all], den_x,
                       out=np.zeros(w, dtype=float), where=den_x != 0)
    wx_all = np.clip(wx_all, 0, 1)
    for i in range(h):
        rb = int(np.searchsorted(row_centers, i, side="right"))
        r0, r1 = max(0, rb - 1), min(nr - 1, rb)
        wy = 0.0 if r0 == r1 else (i - row_centers[r0]) / (row_centers[r1] - row_centers[r0])
        wy = float(np.clip(wy, 0, 1))
        v = img[i]
        top = (1 - wx_all) * luts[r0, c0_all, v] + wx_all * luts[r0, c1_all, v]
        bot = (1 - wx_all) * luts[r1, c0_all, v] + wx_all * luts[r1, c1_all, v]
        out[i] = np.clip(np.rint((1 - wy) * top + wy * bot), 0, 255)
    return out
def metrics(img):
    p = np.bincount(img.ravel(), minlength=256).astype(float)
    p /= p.sum()
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    f = img.astype(float)
    gx = np.diff(f, axis=1)
    gy = np.diff(f, axis=0)
    avg_gradient = float((np.abs(gx).mean() + np.abs(gy).mean()) / 2)
    rms_contrast = float(f.std())
    return entropy, avg_gradient, rms_contrast
def save_gray(path, arr):
    Image.fromarray(arr).save(path)
def main():
    sample = Path(get_sample_data("grace_hopper.jpg", asfileobj=False))
    img = Image.open(sample).convert("L").resize((256, 256), Image.Resampling.LANCZOS)
    original = np.asarray(img, dtype=np.uint8)
    t0 = perf_counter(); lhe = local_hist_equalization(original, 15); t_lhe = perf_counter() - t0
    t0 = perf_counter(); cl = clahe(original, (8, 8), 2.0); t_cl = perf_counter() - t0
    save_gray(OUT / "original.png", original)
    save_gray(OUT / "lhe.png", lhe)
    save_gray(OUT / "clahe.png", cl)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.55), dpi=180)
    for ax, a, title in zip(axes, [original, lhe, cl], ["Original", "LHE (15x15)", "CLAHE (8x8, clip=2.0)"]):
        ax.imshow(a, cmap="gray", vmin=0, vmax=255); ax.set_title(title, fontsize=11); ax.axis("off")
    fig.tight_layout(pad=0.6); fig.savefig(OUT / "comparison.png", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=180)
    for a, label, color in [(original, "Original", "#666666"), (lhe, "LHE", "#d95f02"), (cl, "CLAHE", "#1b78a6")]:
        hist = np.bincount(a.ravel(), minlength=256)
        ax.plot(hist, label=label, color=color, linewidth=1.1)
    ax.set_xlabel("Gray level"); ax.set_ylabel("Pixel count"); ax.grid(alpha=.18); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "histograms.png", bbox_inches="tight"); plt.close(fig)
    rows = []
    for name, a, sec in [("Original", original, 0.0), ("LHE", lhe, t_lhe), ("CLAHE", cl, t_cl)]:
        e, g, c = metrics(a); rows.append((name, e, g, c, sec))
    (OUT / "metrics.csv").write_text("method,entropy,avg_gradient,rms_contrast,time_s\n" +
        "\n".join(f"{n},{e:.4f},{g:.4f},{c:.4f},{s:.4f}" for n,e,g,c,s in rows), encoding="utf-8")
    print((OUT / "metrics.csv").read_text())
if __name__ == "__main__":
    main()
