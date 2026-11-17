# -*- coding: utf-8 -*-
"""武汉理工大学《数字图像处理》实验一：空间域图像增强。
核心直方图、滤波、锐化算法均使用 NumPy 实现；Pillow 仅负责读入/保存；
matplotlib 只负责绘图；skimage 的 equalize_hist 仅作对照，SSIM 用于评价。
运行：python image_experiment1.py
"""
from __future__ import annotations
 
import csv
import json
import time
from pathlib import Path
 
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from skimage.metrics import structural_similarity
from skimage.exposure import equalize_hist  # 仅对照实验，不作为手写均衡化结果
 
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261009)
FONT_FILE = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(FONT_FILE).exists():
    font_manager.fontManager.addfont(FONT_FILE)
    fontname = font_manager.FontProperties(fname=FONT_FILE).get_name()
    plt.rcParams['font.sans-serif'] = [fontname]
else:
    # Windows 优先选择系统已有的中文字体；不依赖 Linux 专用字体路径
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Noto Sans CJK SC']
    for fontname in candidates:
        if fontname in available:
            plt.rcParams['font.sans-serif'] = [fontname]
            break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 160, 'font.size': 9})
 
 
def to_gray(rgb):
    rgb = rgb.astype(np.float32)
    return np.clip(np.rint(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]), 0, 
255).astype(np.uint8)
 
 
def histogram(img):
    return np.bincount(img.ravel(), minlength=256).astype(np.int64)
 
 
def equalize_manual(img):
    hist = histogram(img)
    cdf = np.cumsum(hist, dtype=np.float64)
    nonzero = cdf[hist > 0]
    if len(nonzero) <= 1:  # 常值图像不做增强
        return img.copy(), np.arange(256, dtype=np.uint8)
    cdf0 = nonzero[0]
    lut = np.clip(np.round((cdf - cdf0) / (img.size - cdf0) * 255), 0, 255).astype(np.uint8)
    return lut[img], lut
 
 
def target_histogram():
    x = np.arange(256, dtype=np.float64)
    h = (0.54 * np.exp(-0.5 * ((x - 76) / 22) ** 2)
         + 0.46 * np.exp(-0.5 * ((x - 190) / 25) ** 2))
    return h / h.sum()
 
 
def hist_match_manual(img, target):
    pc = np.cumsum(histogram(img) / img.size)
    pt = np.cumsum(target / np.sum(target))
    lut = np.searchsorted(pt, pc, side='left')
    return np.clip(lut[img], 0, 255).astype(np.uint8), lut.astype(np.uint8)
 
 
def conv2_manual(img, kernel):
    """只沿核偏移进行遍历，不调用图像库内置滤波函数。"""
    x = np.asarray(img, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32)
    ry, rx = k.shape[0] // 2, k.shape[1] // 2
    v = np.pad(x, ((ry, ry), (rx, rx)), mode='reflect')
    out = np.zeros_like(x, dtype=np.float32)
    for iy in range(k.shape[0]):
        for ix in range(k.shape[1]):
            if k[iy, ix] != 0:
                out += k[iy, ix] * v[iy:iy+x.shape[0], ix:ix+x.shape[1]]
    return out
 
 
def normalized_view(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x, dtype=np.uint8) if hi <= lo else np.clip(255 * (x-lo) / (hi-lo), 0, 
255).astype(np.uint8)
 
 
def sharpen_results(img):
    x = img.astype(np.float32)
    k_roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    k_roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    k_sobel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=np.float32)
    k_sobel_y = k_sobel_x.T
    # Roberts 2x2 采用窗口左上角锚点以对齐图像
    z = np.pad(x, ((0, 1),(0, 1)), mode='reflect')
    rx = z[:-1,:-1] - z[1:,1:]
    ry = z[:-1,1:] - z[1:,:-1]
    rob = np.sqrt(rx*rx + ry*ry)
    sx, sy = conv2_manual(x, k_sobel_x), conv2_manual(x, k_sobel_y)
    sob = np.sqrt(sx*sx + sy*sy)
    k_prewitt_x = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]], dtype=np.float32)
    px, py = conv2_manual(x, k_prewitt_x), conv2_manual(x, k_prewitt_x.T)
    prewitt = np.sqrt(px*px + py*py)
    lap = conv2_manual(x, [[0,1,0],[1,-4,1],[0,1,0]])
    return {
        'roberts': normalized_view(rob),
        'sobel': normalized_view(sob),
        'prewitt': normalized_view(prewitt),
        'sobel_sharp': np.clip(x + 0.30 * sob, 0, 255).astype(np.uint8),
        'lap_abs': np.clip(np.abs(lap),0,255).astype(np.uint8),
        'lap_stretch': normalized_view(lap),
        'lap_overlay': np.clip(x - lap, 0, 255).astype(np.uint8),
        'lap_raw': lap,
    }
 
 
def box_mean(img, k):
    """积分图法：k×k 盒式均值，边界镜像延拓。"""
    assert k % 2 == 1 and k > 0
    r = k//2
    p = np.pad(np.asarray(img, dtype=np.float64), ((r,r),(r,r)), mode='reflect')
    s = np.pad(np.cumsum(np.cumsum(p, axis=0),axis=1), ((1,0),(1,0)), mode='constant')
    return ((s[k:,k:] - s[:-k,k:] - s[k:,:-k] + s[:-k,:-k])/(k*k)).astype(np.float32)
 
 
def mean_filter(img, k=3, center_weight=1.0):
    if center_weight == 1.0:
        return box_mean(img, k)
    w = np.ones((k,k),dtype=np.float32)
    w[k//2,k//2] = float(center_weight)
    return conv2_manual(img, w/w.sum())
 
 
def median_filter(img, k=3):
    x = np.asarray(img, dtype=np.float32)
    r = k//2
    p = np.pad(x, ((r,r),(r,r)), mode='reflect')
    patches = np.stack([p[j:j+x.shape[0],i:i+x.shape[1]] for j in range(k) for i in range(k)])
    return np.median(patches,axis=0).astype(np.float32)
 
 
def gaussian_filter(img, k=5, sigma=1.0):
    ax = np.arange(-(k//2), k//2+1, dtype=np.float32)
    g = np.exp(-ax**2/(2*sigma*sigma))
    g /= np.sum(g)
    return conv2_manual(img, np.outer(g,g))
 
 
def bilateral_filter(img, k=7, sigma_space=2.0, sigma_range=25.0):
    x = np.asarray(img,dtype=np.float32)
    r = k//2
    padded = np.pad(x, ((r,r),(r,r)),mode='reflect')
    weighted = np.zeros_like(x)
    norm = np.zeros_like(x)
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            neigh = padded[dy+r:dy+r+x.shape[0],dx+r:dx+r+x.shape[1]]
            weight = np.exp(-(dx*dx+dy*dy)/(2*sigma_space**2) - (neigh-x)**2/(2*sigma_range**2))
            weighted += weight*neigh
            norm += weight
    return weighted / np.maximum(norm,1e-7)
 
 
def guided_filter(img, radius=4, eps=0.012):
    """自引导滤波：输入和引导图相同，计算在 [0,1] 尺度上进行。"""
    I = np.asarray(img,dtype=np.float32)/255.0
    k = 2*radius+1
    mu = box_mean(I,k)
    mu2 = box_mean(I*I,k)
    var = np.maximum(mu2 - mu*mu,0)
    a = var/(var+eps)
    b = mu-a*mu
    out = box_mean(a,k)*I + box_mean(b,k)
    return np.clip(out*255,0,255)
 
 
def nlm_filter(img, search_radius=5, patch_size=3, h=17, sigma_noise=16):
    """手写 NLM：空间移位 + patch SSD 盒滤波，排除中心后由局部最大权回填。"""
    x = np.asarray(img,dtype=np.float32)
    p = np.pad(x, ((search_radius,search_radius),(search_radius,search_radius)),mode='reflect')
    acc = np.zeros_like(x)
    norm = np.zeros_like(x)
    max_w = np.zeros_like(x)
    for dy in range(-search_radius,search_radius+1):
        for dx in range(-search_radius,search_radius+1):
            if dy==0 and dx==0: continue
            shifted = p[search_radius+dy:search_radius+dy+x.shape[0],
                        search_radius+dx:search_radius+dx+x.shape[1]]
            distance = box_mean((x-shifted)**2,patch_size)
            # 去除白噪声方差的期望贡献，防止权重退化
            distance = np.maximum(distance-2*sigma_noise**2,0)
            w = np.exp(-distance/(h*h)).astype(np.float32)
            acc += w*shifted
            norm += w
            max_w = np.maximum(max_w,w)
    acc += max_w*x
    norm += max_w
    return acc/np.maximum(norm,1e-7)
 
 
def uint8(img):
    return np.clip(np.rint(img),0,255).astype(np.uint8)
 
 
def gaussian_noise(img, sigma=16):
    return uint8(img.astype(np.float32)+RNG.normal(0,sigma,img.shape))
 
 
def salt_pepper_noise(img, p=0.035):
    out=img.copy()
    noise=RNG.random(img.shape)
    out[noise<p/2]=0
    out[(noise>=p/2)&(noise<p)]=255
    return out
 
 
def psnr(ref,img):
    mse=np.mean((ref.astype(np.float64)-np.asarray(img,dtype=np.float64))**2)
    return float('inf') if mse==0 else float(10*np.log10((255**2)/mse))
 
 
def ssim(ref,img):
    return float(structural_similarity(ref,uint8(img),data_range=255))
 
 
def draw_img(ax,img,title):
    ax.imshow(img,cmap='gray',vmin=0,vmax=255)
    ax.set_title(title,pad=6)
    ax.axis('off')
 
 
def save_fig(fig,name):
    fig.savefig(OUT/name,bbox_inches='tight',facecolor='white')
    plt.close(fig)
 
 
def run(input_path=None):
    """Reconstructed entrypoint from PDF appendix; requires 512x512 input image."""
    image_path=Path(input_path) if input_path else ROOT/'face_input.jpg'
    if not image_path.exists():
        raise FileNotFoundError(f"Missing original image: {image_path}; pass --input /path/to/face.jpg")
    image=to_gray(np.asarray(Image.open(image_path).convert('RGB')))
    if image.shape!=(512,512):
        image=np.asarray(Image.fromarray(image).resize((512,512), Image.Resampling.LANCZOS))
    low=uint8(0.47*image.astype(np.float32)+12)
    eq,lut_eq=equalize_manual(low)
    target=target_histogram()
    matched,lut_mat=hist_match_manual(low,target)
    lib_eq=uint8(255*equalize_hist(low,nbins=256))
    sharp=sharpen_results(image)
    noise_g=gaussian_noise(image,16)
    noise_sp=salt_pepper_noise(image,0.04)
    noise_mixed=salt_pepper_noise(gaussian_noise(image,16),0.025)
    for name,arr in [('original',image),('dark_control',low),('hist_equalized',eq),
                     ('hist_matched',matched),('gaussian_noisy',noise_g),
                     ('saltpepper_noisy',noise_sp),('mixed_noisy',noise_mixed)]:
        Image.fromarray(uint8(arr)).save(OUT/(name+'.png'))
    fig,axs=plt.subplots(2,3,figsize=(10,6.4))
    variants=[(image,'Original'),(low,'Darkened'),(eq,'Manual equalization'),
              (matched,'Histogram specification'),(lib_eq,'Library comparison'),
              (np.abs(eq.astype('int16')-lib_eq.astype('int16')),'Absolute difference')]
    for ax,(arr,name) in zip(axs.flat,variants): draw_img(ax,arr,name)
    fig.tight_layout();save_fig(fig,'fig01_hist_images.png')
    fig,axs=plt.subplots(2,3,figsize=(10.8,5.2))
    configs=[(low,'Input histogram'),(eq,'Equalized histogram'),(matched,'Specified histogram')]
    for i,(arr,name) in enumerate(configs):
        axs[0,i].bar(np.arange(256),histogram(arr)/arr.size,width=1.0)
        axs[0,i].set(title=name,xlabel='Gray level',ylabel='Frequency',xlim=(0,255))
    axs[1,0].plot(np.arange(256),np.cumsum(histogram(low))/low.size,label='Input CDF')
    axs[1,0].plot(np.arange(256),np.cumsum(target),label='Target CDF');axs[1,0].legend()
    axs[1,1].plot(lut_eq,label='Equalization LUT');axs[1,1].plot(lut_mat,label='Specification LUT');axs[1,1].legend()
    axs[1,2].plot(target,label='Target');axs[1,2].plot(histogram(matched)/matched.size,label='Output');axs[1,2].legend()
    fig.tight_layout();save_fig(fig,'fig02_hist_charts.png')
    fig,axs=plt.subplots(2,4,figsize=(11,6))
    items=[(image,'Original'),(sharp['roberts'],'Roberts'),(sharp['sobel'],'Sobel'),
           (sharp['sobel_sharp'],'Sobel enhanced'),(sharp['lap_abs'],'Laplacian abs'),
           (sharp['lap_stretch'],'Laplacian stretched'),(sharp['lap_overlay'],'Laplacian enhanced'),
           (sharp['prewitt'],'Prewitt')]
    for ax,(arr,name) in zip(axs.flat,items): draw_img(ax,arr,name)
    fig.tight_layout();save_fig(fig,'fig03_edges.png')
    settings={
        'mean':lambda a:mean_filter(a,3),
        'median':lambda a:median_filter(a,3),
        'gaussian':lambda a:gaussian_filter(a,5,1.1),
        'bilateral':lambda a:bilateral_filter(a,7,2.,25),
        'guided':lambda a:guided_filter(a,4,.012),
        'NLM':lambda a:nlm_filter(a,5,3,17,16)}
    records=[]; results={}
    for noise_label,noisy in [('Gaussian',noise_g),('SaltPepper',noise_sp),('Mixed',noise_mixed)]:
        records.append(dict(noise=noise_label,algorithm='Input',psnr=psnr(image,noisy),ssim=ssim(image,noisy),seconds=0.))
        for name,fn in settings.items():
            tic=time.perf_counter(); y=uint8(fn(noisy)); seconds=time.perf_counter()-tic
            records.append(dict(noise=noise_label,algorithm=name,psnr=psnr(image,y),ssim=ssim(image,y),seconds=seconds))
            results[(noise_label,name)]=y
    fig,axs=plt.subplots(2,4,figsize=(12,6.4))
    pics=[(image,'Original'),(noise_mixed,'Mixed noise')]+[(results[('Mixed',name)],name) for name in settings]
    for ax,(arr,name) in zip(axs.flat,pics): draw_img(ax,arr,name)
    fig.tight_layout();save_fig(fig,'fig04_mixed_denoise.png')
    import csv
    with (OUT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['noise','algorithm','psnr','ssim','seconds'])
        writer.writeheader();writer.writerows(records)
    summary=dict(seed=20261009,input_shape=list(image.shape),
                 manual_vs_library_mae=float(np.mean(np.abs(eq.astype(float)-lib_eq.astype(float)))),
                 note='Rebuilt entrypoint; historical report metrics depend on original input image.')
    (OUT/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default=None,help='Path to source portrait/image; original used face_input.jpg')
    args=parser.parse_args()
    run(args.input)
