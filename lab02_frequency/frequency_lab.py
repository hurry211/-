# -*- coding: utf-8 -*-
"""实验二 图像增强的频域方法 | Python 3.10+
核心滤波器与 DFT 均通过数学公式实现，不调用现成低通、高通、陷波滤波器。
运行：python 实验二_频域增强_完整代码.py
交互陷波：python 实验二_频域增强_完整代码.py --interactive 16.jpg
依赖：numpy、scipy、matplotlib、Pillow、scikit-image（仅使用其示例照片，不用于滤波）
提供图像：7.PNG、10.jpg、11.jpg、16.jpg；路径为代码同级的 input_images 文件夹。
"""
import argparse
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib
# 命令行交互模式：必须在导入 pyplot 之前选择图形后端。
import sys
matplotlib.use('TkAgg' if '--interactive' in sys.argv else 'Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image
from skimage import data
 
BASE = Path(__file__).resolve().parent
INPUT = BASE / 'input_images'
OUTPUT = BASE / 'results'
OUTPUT.mkdir(exist_ok=True)
SEED = 20261106
LOW_D0 = (5, 15, 30, 60)
HIGH_D0 = (5, 15, 30, 60)
FILTERS = ('Ideal', 'Butterworth', 'Gaussian')
# 调整时仅更新此参数表，坐标均为中心化 FFT 中的频率偏移 (du,dv)。
# 实际图片缺乏未受噪声污染真值，故半径通过频谱峰宽/图像视觉效果选取。
NOTCH_D0 = {'7.PNG': 2.6, '10.jpg': 2.5, '11.jpg': 2.0, '16.jpg': 3.2}
 
 
def set_plot_style():
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP','DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 110
    plt.rcParams['savefig.dpi'] = 155
 
 
def save(fig, filename):
    fig.savefig(OUTPUT/filename, bbox_inches='tight', facecolor='white')
    plt.close(fig)
 
 
def gray01(image):
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = 0.299*arr[:,:,0]+0.587*arr[:,:,1]+0.114*arr[:,:,2]
    arr = arr.astype(np.float64)
    if np.max(arr) > 1.0:
        arr /= 255.0
    return np.clip(arr, 0., 1.)
 
 
def center_resize(image, shape=(256,256)):
    im = Image.fromarray(np.uint8(np.round(gray01(image)*255)),mode='L')
    width,height=im.size; side=min(width,height)
    x,y=(width-side)//2,(height-side)//2
    im=im.crop((x,y,x+side,y+side))
    im=im.resize((shape[1],shape[0]), Image.Resampling.LANCZOS)
    return gray01(im)
 
 
def log_spectrum(F):
    z=np.log1p(np.abs(F))
    return (z-z.min())/(z.max()-z.min()+1e-12)
 
 
def dft2_manual(image):
    """二维 DFT 的可分离矩阵形式，未调用 np.fft.*。"""
    n,m=image.shape
    u=np.arange(n,dtype=np.float64)
    v=np.arange(m,dtype=np.float64)
    Wn=np.exp((-2j*np.pi/n) * np.outer(u,u))
    Wm=np.exp((-2j*np.pi/m) * np.outer(v,v))
    return Wn @ image.astype(np.complex128) @ Wm.T
 
 
def fft_filter(image, H):
    F=np.fft.fftshift(np.fft.fft2(image))
    g=np.fft.ifft2(np.fft.ifftshift(F*H))
    return np.real(g)
 
 
def distance_grid(shape):
    yy,xx=np.indices(shape)
    return np.hypot(yy-shape[0]//2,xx-shape[1]//2)
 
 
def transfer(shape, family, mode, D0, order=2):
    """用频域传递函数从零构建六种滤波器。"""
    if D0<=0: raise ValueError('D0 必须大于0')
    D=distance_grid(shape)
    if family=='Ideal':
        low=np.float64(D<=D0)
    elif family=='Butterworth':
        low=1./(1.+np.power(D/D0,2*order))
    elif family=='Gaussian':
        low=np.exp(-D**2/(2*D0**2))
    else: raise ValueError('未知滤波器: '+str(family))
    if mode=='low': return low
    if mode=='high': return 1.-low
    raise ValueError('mode 应为low 或high')
 
 
def psnr(reference, estimation, max_val=1.0):
    mse=np.mean((np.asarray(reference)-np.asarray(estimation))**2)
    return np.inf if mse==0 else 10*np.log10(max_val**2/mse)
 
 
def part1():
    f=np.zeros((256,256),np.float64)
    f[120:136,126:130]=1.0   # 精确16x4，中心在 (127.5, 127.5)
    F=np.fft.fft2(f)
    G=np.fft.fftshift(F)
    manual=dft2_manual(f)
    err=np.max(np.abs(manual-F))
    restored=np.fft.ifftshift(G)
    assert np.allclose(restored,F,atol=1e-11)
    assert err < 1e-9, err
    fig,ax=plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
    for a,im,title in zip(ax.flat,
      (f,log_spectrum(F),np.angle(F),log_spectrum(G),np.angle(G),log_spectrum(restored)),
      ('16×4 白块（256×256）','幅度谱（未中心化）','相位谱（未中心化）',
       '幅度谱 fftshift 后','相位谱 fftshift 后','ifftshift 复原频谱')):
        a.imshow(im,cmap='gray' if '相位' not in title else 'twilight',interpolation='nearest')
        a.set_title(title,fontsize=11);a.axis('off')
    fig.suptitle(f'傅里叶变换与频谱平移（手写 DFT 最大误差={err:.2e}）',fontsize=14)
    save(fig,'01_FFT_幅度相位与平移.png')
    return {'manual_fft_max_abs_error':float(err),'ifftshift_reconstruction_error':float(np.max(np.abs(restored-F)))}
 
 
def part2_lowpass():
    clean=center_resize(data.camera())
    rng=np.random.default_rng(SEED)
    noisy=np.clip(clean+rng.normal(0,0.08,clean.shape),0,1)
    np.save(OUTPUT/'lowpass_clean.npy',clean)
    np.save(OUTPUT/'lowpass_noisy.npy',noisy)
    metrics=[]
    fig,ax=plt.subplots(2,2,figsize=(8,7),constrained_layout=True)
    for a,im,title in zip(ax.flat,(clean,noisy,log_spectrum(np.fft.fftshift(np.fft.fft2(clean))),
                     log_spectrum(np.fft.fftshift(np.fft.fft2(noisy)))),
         ('原始实景灰度照片','加入 Gaussian N(0, 0.08²) 噪声','原始图像幅度谱','加噪图像幅度谱')):
        a.imshow(im,cmap='gray',vmin=0,vmax=1);a.set_title(title,fontsize=11);a.axis('off')
    fig.suptitle(f'低通实验输入与频谱：加噪 PSNR={psnr(clean,noisy):.2f} dB')
    save(fig,'02_原始图_加噪图_频谱.png')
    fig,axes=plt.subplots(3,4,figsize=(12,9),constrained_layout=True)
    best=None
    for i,fam in enumerate(FILTERS):
        for j,D0 in enumerate(LOW_D0):
            out=fft_filter(noisy,transfer(noisy.shape,fam,'low',D0))
            show=np.clip(out,0,1)
            val=psnr(clean,show)
            metrics.append({'filter':fam,'D0':D0,'PSNR_dB':float(val)})
            if best is None or val>best['PSNR_dB']:best=metrics[-1]
            a=axes[i,j];a.imshow(show,cmap='gray',vmin=0,vmax=1)
            a.set_title(f'{fam}, D0={D0}\nPSNR={val:.2f} dB',fontsize=9);a.axis('off')
    fig.suptitle('3 类低通滤波器 × 4 种截止频率（同一含噪图像）',fontsize=14)
    save(fig,'03_低通滤波_3x4 对照.png')
    return metrics,{'noisy_psnr':float(psnr(clean,noisy)),'best':best}
 
 
def part2_highpass():
    clean=center_resize(data.astronaut())
    fig,axes=plt.subplots(3,4,figsize=(12,9),constrained_layout=True)
    metrics=[]
    for i,fam in enumerate(FILTERS):
        for j,D0 in enumerate(HIGH_D0):
            Hp=fft_filter(clean,transfer(clean.shape,fam,'high',D0))
            # 高频结果是正负信号；展示其绝对值而非把它伪装为自然图像。
            view=np.abs(Hp)
            scale=np.quantile(view,0.995)
            axes[i,j].imshow(np.clip(view/(scale+1e-12),0,1),cmap='gray',vmin=0,vmax=1)
            axes[i,j].set_title(f'{fam}, D0={D0}',fontsize=10);axes[i,j].axis('off')
            metrics.append({'filter':fam,'D0':D0,'signed_rms':float(np.sqrt(np.mean(Hp**2)))})
    fig.suptitle('3 类高通滤波器 × 4 种截止频率（显示 |高频响应|）',fontsize=14)
    save(fig,'04_高通滤波_3x4 对照.png')
    fig,ax=plt.subplots(1,4,figsize=(13,3.8),constrained_layout=True)
    ax[0].imshow(clean,cmap='gray',vmin=0,vmax=1);ax[0].set_title('原始自然照片')
    configs=[('Ideal',10,0.9),('Butterworth',12,1.4),('Gaussian',12,1.4)]
    sharp_stats=[]
    for a,(fam,D0,k) in zip(ax[1:],configs):
        Hp=fft_filter(clean,transfer(clean.shape,fam,'high',D0))
        enhanced=np.clip(clean+k*Hp,0,1)
        a.imshow(enhanced,cmap='gray',vmin=0,vmax=1)
        a.set_title(f'{fam} 高频增强\nD0={D0}, k={k}')
        sharp_stats.append({'filter':fam,'D0':D0,'gain':k})
    for a in ax:a.axis('off')
    fig.suptitle('高频增强 g=clip(f+k·HP(f),0,1)，高斯/巴特沃斯边缘更自然',fontsize=12)
    save(fig,'05_高通增强结果.png')
    return metrics,sharp_stats
 
 
def ringing_experiment():
    f=np.zeros((256,256));f[64:192,64:192]=1.0
    D0=18
    ideal=fft_filter(f,transfer(f.shape,'Ideal','low',D0))
    gauss=fft_filter(f,transfer(f.shape,'Gaussian','low',D0))
    bw=fft_filter(f,transfer(f.shape,'Butterworth','low',D0))
    fig,ax=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for a,g,title in zip(ax[0],[f,ideal,gauss],['原始阶跃方块','理想低通：波纹明显','高斯低通：边缘平缓']):
        a.imshow(g,cmap='gray',vmin=0,vmax=1);a.set_title(title);a.axis('off')
    xx=np.arange(256)
    for g,t in [(f,'原始'),(ideal,'理想低通'),(bw,'巴特沃斯'),(gauss,'高斯')]:
        ax[1,0].plot(xx,g[128,:],label=t,linewidth=1.1)
    ax[1,0].set_xlim(35,105);ax[1,0].set_ylim(-.15,1.25);ax[1,0].set_title('水平剖面的过冲与振铃')
    ax[1,0].grid(alpha=.25);ax[1,0].legend(fontsize=8)
    ax[1,1].imshow(bw,cmap='gray',vmin=0,vmax=1);ax[1,1].set_title('巴特沃斯低通');ax[1,1].axis('off')
    ax[1,2].imshow(np.abs(ideal-gauss),cmap='magma');ax[1,2].set_title('理想与高斯滤波差异绝对值');ax[1,2].axis('off')
    fig.suptitle('同截止频率 D0=18 的振铃效应比较',fontsize=14)
    save(fig,'06_理想低通振铃机理.png')
    return {'Ideal_min':float(ideal.min()),'Ideal_max':float(ideal.max()),
      'Gaussian_min':float(gauss.min()),'Gaussian_max':float(gauss.max())}
 
 
def symmetric_centers(shape, centers):
    """中心化频率坐标(u,v)：横轴u，纵轴v，与其负坐标成对。"""
    h,w=shape;out=[]
    for du,dv in centers:
        if (du,dv)==(0,0):continue
        out.extend(((w//2+du,h//2+dv),(w//2-du,h//2-dv)))
    return out
 
 
def notch_transfer(shape, centers, D0, order=2):
    """成对巴特沃斯陷波拒绝滤波器，中心点为(du,dv)，防止破坏实数对称。"""
    h,w=shape;yy,xx=np.indices(shape)
    H=np.ones(shape,np.float64)
    for du,dv in centers:
        if du==0 and dv==0:continue
        d1=np.hypot(xx-(w//2+du),yy-(h//2+dv))
        d2=np.hypot(xx-(w//2-du),yy-(h//2-dv))
        # 可消除两侧成对噪声谱峰，d=0 时令透过率0。
        q1=np.where(d1>0,1./(1.+(D0/np.maximum(d1,1e-12))**(2*order)),0.)
        q2=np.where(d2>0,1./(1.+(D0/np.maximum(d2,1e-12))**(2*order)),0.)
        H*=q1*q2
    return H
 
 
def detect_peaks(img, max_pairs=12, exclusion=9, min_separation=7):
    """针对相对频谱背景突起的局部峰值筛选，并只保留半平面的一半以成对陷波。"""
    F=np.fft.fftshift(np.fft.fft2(img)); L=np.log1p(np.abs(F))
    # 去除自然图像的缓变频谱趋势，突出狭窄周期峰。
    whiten=L-ndimage.gaussian_filter(L,sigma=5.0)
    h,w=img.shape;yy,xx=np.indices(img.shape)
    rr=np.hypot(xx-w//2,yy-h//2)
    peaks=(whiten==ndimage.maximum_filter(whiten,size=7,mode='wrap')) & (rr>exclusion)
    peaks &= (rr<min(h,w)*.47)
    scores=np.where(peaks,whiten,-np.inf)
    indices=np.argsort(scores.ravel())[::-1]
    centers=[]; strength=[]
    threshold=max(0.75,float(np.quantile(whiten[rr>exclusion],0.997)))
    for ind in indices:
        y,x=np.unravel_index(ind,img.shape)
        score=scores[y,x]
        if not np.isfinite(score) or score < threshold: break
        du=int(x-w//2);dv=int(y-h//2)
        # 每对(du,dv)只记录频率空间的一个半平面。
        if dv<0 or (dv==0 and du<0):du,dv=-du,-dv
        if any(np.hypot(du-cx,dv-cy)<min_separation or np.hypot(du+cx,dv+cy)<min_separation for cx,cy in centers):
            continue
        centers.append((du,dv));strength.append(float(score))
        if len(centers)>=max_pairs:break
    return centers,strength,whiten
 
 
def notch_image(img, centers, D0):
    H=notch_transfer(img.shape,centers,D0)
    return np.clip(fft_filter(img,H),0,1),H
 
 
def challenge(images=None):
    if images is None: images=('7.PNG','10.jpg','11.jpg','16.jpg')
    metrics=[]
    fig,axes=plt.subplots(len(images),4,figsize=(13,3.05*len(images)),constrained_layout=True)
    for row,name in enumerate(images):
        img=gray01(Image.open(INPUT/name))
        centers,scores,whiten=detect_peaks(img,max_pairs=12)
        D0=NOTCH_D0.get(name,2.5)
        filtered,H=notch_image(img,centers,D0)
        original_sp=log_spectrum(np.fft.fftshift(np.fft.fft2(img)))
        draw_sp=original_sp.copy()
        save_out=(OUTPUT/('07_陷波_'+Path(name).stem+'.png'))
        Image.fromarray(np.uint8(np.round(filtered*255))).save(save_out)
        for col,im,title in zip(range(4),
                   (img,original_sp,filtered,log_spectrum(np.fft.fftshift(np.fft.fft2(filtered)))),
                   (f'{name} 原始图',f'频谱/检测 {len(centers)} 对峰',f'陷波结果 D0={D0}', '陷波后频谱')):
            a=axes[row,col];a.imshow(im,cmap='gray',vmin=0,vmax=1)
            a.set_title(title,fontsize=10);a.axis('off')
        for du,dv in centers:
            for sx,sy in symmetric_centers(img.shape,[(du,dv)]):
                axes[row,1].plot(sx,sy,'ro',ms=3,fillstyle='none',mew=.9)
        metrics.append({'image':name,'shape':list(img.shape),'D0':D0,
                        'notch_pairs':centers,'peak_scores':[round(x,3) for x in scores],
                        'relative_change_percent':float(100*np.sqrt(np.mean((img-filtered)**2))/np.sqrt(np.mean(img**2))),
                        'PSNR_vs_clean':'不可计算：未提供真值'})
    fig.suptitle('附加挑战：自动定位周期谱峰 + 成对巴特沃斯陷波（原图无真值）',fontsize=14)
    save(fig,'08_四张周期噪声图_陷波处理对照.png')
    return metrics
 
 
def synthetic_notch_benchmark():
    """独立构造有真值的周期噪声，用真 PSNR 评价最佳陷波半径。"""
    clean=center_resize(data.camera())
    yy,xx=np.indices(clean.shape)
    periodic=.10*np.cos(2*np.pi*(13*xx+29*yy)/256)+.075*np.cos(2*np.pi*(33*xx-17*yy)/256)
    noisy=np.clip(clean+periodic,0,1)
    centers=[(13,29),(33,-17)]
    radii=(0.8,1.2,1.8,2.5,3.5,5.0,7.0,9.0)
    trials=[]
    for D0 in radii:
        out,_=notch_image(noisy,centers,D0)
        trials.append((float(D0),float(psnr(clean,out))))
    best_r,best_p=max(trials,key=lambda x:x[1])
    bestimg,_=notch_image(noisy,centers,best_r)
    fig,ax=plt.subplots(1,4,figsize=(13,4),constrained_layout=True)
    for a,g,t in zip(ax[:3],(clean,noisy,bestimg),('真实无噪图',f'人工周期噪声 PSNR {psnr(clean,noisy):.2f} dB',
                                   f'最优半径 D0={best_r} PSNR {best_p:.2f} dB')):
        a.imshow(g,cmap='gray',vmin=0,vmax=1);a.set_title(t,fontsize=10);a.axis('off')
    ax[3].plot([x for x,y in trials],[y for x,y in trials],marker='o')
    ax[3].axvline(best_r,color='gray',ls='--');ax[3].set_xlabel('D0（像素）');ax[3].set_ylabel('PSNR (dB)')
    ax[3].grid(alpha=.25);ax[3].set_title('有真值条件下 PSNR 寻优')
    fig.suptitle('模拟周期噪声：已知真值，客观比较陷波半径',fontsize=13)
    save(fig,'09_模拟周期噪声_PSNR 寻优.png')
    return {'baseline_psnr':float(psnr(clean,noisy)),'best_D0':best_r,'best_psnr':best_p,'trials':trials}
 
 
def interactive_notch(name):
    """本机图形界面使用：点击中心化频谱上的干扰峰（只点一侧），键盘Enter 结束选点。
    然后拖动半径滑块实时观察输出，点击“保存结果”保存当前结果。
    注意不要点击 DC 中心亮斑 (cx,cy)。
    """
    plt_gui = plt
    img=gray01(Image.open(INPUT/name)); h,w=img.shape
    F=np.fft.fftshift(np.fft.fft2(img))
    fig,ax=plt_gui.subplots(figsize=(7,6))
    ax.imshow(log_spectrum(F),cmap='gray');ax.set_title('在干扰谱峰上点击一侧，按 Enter 结束')
    ax.scatter([w//2],[h//2],c='y',s=20,marker='+')
    picked=plt_gui.ginput(n=-1,timeout=0)
    plt_gui.close(fig)
    centers=[]
    for x,y in picked:
        du=int(round(x-w//2));dv=int(round(y-h//2))
        if du==dv==0:continue
        if dv<0 or (dv==0 and du<0):du,dv=-du,-dv
        if (du,dv) not in centers:centers.append((du,dv))
    if not centers:centers,_,_=detect_peaks(img)
    fig,ax=plt_gui.subplots(1,2,figsize=(10,5))
    plt_gui.subplots_adjust(bottom=.22)
    ax[0].imshow(img,cmap='gray',vmin=0,vmax=1);ax[0].set_title('原图')
    init=NOTCH_D0.get(name,2.5)
    filtered,_=notch_image(img,centers,init)
    display=ax[1].imshow(filtered,cmap='gray',vmin=0,vmax=1)
    ax[1].set_title('滤波结果：拖动滑块设置半径')
    for a in ax:a.axis('off')
    slider_ax=fig.add_axes([.22,.10,.52,.035]);slider=Slider(slider_ax,'D0',.4,12,valinit=init,valstep=.2)
    button_ax=fig.add_axes([.42,.025,.18,.05]);button=Button(button_ax,'保存结果')
    def refresh(v):
        im,_=notch_image(img,centers,float(v));display.set_data(im);fig.canvas.draw_idle()
    def save_current(event):
        im,_=notch_image(img,centers,float(slider.val))
        file=OUTPUT/('交互陷波_'+Path(name).stem+'.png')
        Image.fromarray(np.uint8(np.round(im*255))).save(file)
        print('已保存:',file,'半径:',slider.val,'中心:',centers)
    slider.on_changed(refresh);button.on_clicked(save_current)
    plt_gui.show()
 
 
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--interactive',choices=['7.PNG','10.jpg','11.jpg','16.jpg'],default=None)
    args=parser.parse_args()
    set_plot_style()
    if args.interactive:
        interactive_notch(args.interactive);return
    import json
    a=part1();low,low_sum=part2_lowpass()
    high,sharp=part2_highpass()
    ring=ringing_experiment()
    available=[name for name in ('7.PNG','10.jpg','11.jpg','16.jpg') if (INPUT/name).exists()]
    notch=challenge(available) if len(available)==4 else []
    if len(available)!=4:
        print('Skip four-image notch challenge: input_images/ originals were not recovered.')
    synth=synthetic_notch_benchmark()
    allmetrics={'part1':a,'lowpass':low,'lowpass_summary':low_sum,
     'highpass':high,'highpass_enhanced':sharp,'ringing':ring,
     'challenge':notch,'synthetic_notch':synth,'random_seed':SEED}
    (OUTPUT/'metrics.json').write_text(json.dumps(allmetrics,ensure_ascii=False,indent=2),encoding='utf8')
    print('图像生成完成。低通最佳：',low_sum['best'],'| 模拟陷波最佳：',synth['best_D0'],synth['best_psnr'])
    print('所有图片和指标写入：',OUTPUT)
 
if __name__=='__main__':
    main()
