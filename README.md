# Digital Image Processing Labs | 数字图像处理实验

武汉理工大学数学与统计学院数字图像处理课程实验的**代码整理与可复现版本**，涵盖空域增强、LHE/CLAHE 和频域增强。核心目的是展示从数学公式到 Python 实现、参数控制、实验评测与失败分析的过程。

> **来源和真实性说明**：三个章节均依据2026年课程实验报告附录恢复；仓库代码是从PDF恢复并经过必要修复的版本，**不是声称与遗失的原始 `.py` 文件逐字节相同**。每个章节的源文件、改动及重运行说明见 `PROVENANCE.md`。本次用内置/替代图像运行产生的数值，不能代替原报告的人脸图像及四张周期噪声图结果。

## 实验目录

| 目录 | 主题 | 主要方法 | 当前状态 |
|---|---|---|---|
| `lab00_local_hist` | LHE / CLAHE | 局部CDF、直方图限幅、双线性插值 | 从PDF附录恢复，已在当前环境重运行 |
| `lab01_spatial` | 图像增强的空域方法 | 均衡化、规定化、Roberts/Sobel/Prewitt/Laplacian、均值/中值/高斯/双边/引导/NLM滤波 | 核心函数从PDF恢复；主程序受分页换行影响已重构；在替代输入上通过测试 |
| `lab02_frequency` | 图像增强的频域方法 | DFT/FFT、幅度与相位、fftshift、理想/巴特沃斯/高斯低高通、振铃、成对陷波、PSNR | 从PDF附录恢复，已在当前环境重运行；缺失四张原始周期噪声图的部分自动跳过 |

| `lab03_motion_psf` | MATLAB motion blur PSF | `my_motion_psf.m`, `test_motion_psf.m` | 历史报告与独立`.m`文件已找回；未在本次环境重新运行MATLAB |

## 环境与运行

Python 3.10+，建议建立虚拟环境：

```bash
python -m pip install -r requirements.txt
python lab00_local_hist/lhe_clahe_lab.py
python lab02_frequency/frequency_lab.py
python -m pytest -q
```

空域实验使用报告中的 `face_input.jpg`（**原始图像尚未找回**）。可准备一张有授权的灰度或RGB人像作为替代：

```bash
python lab01_spatial/spatial_lab.py --input path/to/your_image.jpg
```

频域实验的周期噪声扩展部分使用 `lab02_frequency/input_images/7.PNG`、`10.jpg`、`11.jpg`、`16.jpg`；没有这些文件时，程序会如实提示并跳过该子任务。其他模拟实验仍可运行。

部分交互式陷波实验需要本地带有 Tk 支持的图形桌面：

```bash
python lab02_frequency/frequency_lab.py --interactive 11.jpg
```

输出默认写到各实验目录下的 `results/` 或 `figures/`，由 `.gitignore` 排除，以避免将每次运行结果误标为统一基准。

## 当前重新运行的验证（非历史原图结果）

- LHE / CLAHE：使用 Matplotlib 自带 Grace Hopper 照片（256×256）；算法正常输出对比图与 `metrics.csv`。本次重运行局部均衡化信息熵为 **7.7711 bit**，CLAHE为 **7.5071 bit**。
- 频域实验：256×256 中心16×4白块，手写矩阵DFT与 `fft2` 最大绝对误差 **5.3608×10⁻¹²**；`ifftshift` 频谱还原误差 **0**。
- 频域模拟加噪低通：高斯低通 `D0=60` 的有真值 PSNR约为 **27.246 dB**；合成周期噪声的最佳试验半径 `D0=1.2`，PSNR约 **37.491 dB**。这些指标只针对程序内构造的固定种子数据，不是四张原始噪声图片的PSNR。
- 空域实验：使用 `skimage.data.camera()` 作为**替代输入**完成本次运行；因为并非原始人像，历史报告中的人脸指标**不可直接比较**。

## 数学与方法

- 图像变换：二维DFT、幅度与相位、中心化频率坐标；手写实现用于与 NumPy FFT 验证一致性。
- 频域滤波：理想低通、二阶巴特沃斯低通、高斯低通及其对应高通，研究截止频率与振铃。
- 图像退化与鲁棒性：周期噪声成对谱峰陷波；有真值模拟PSNR与真实无真值图像的视觉评价严格区分。
- 空间域增强：CDF与查找表映射、离散卷积算子、边缘响应、多种平滑算法及PSNR/SSIM指标。

## 研究记录与局限

- `.py` 源码按报告附录复原，修复PDF换行、HTML转义字符，以及临时输出路径；详细记录在 `PROVENANCE.md`。
- 空域实验的主程序是根据报告的原始算法/输出项**重建的入口**，不是完整原生脚本逐字恢复。
- 频域实验缺失四张原始图像，不能复现该小节的最终去噪图，且不能把合成周期噪声结果当作真实图像结果。
- LHE对噪声敏感且较慢；CLAHE的限幅机制和双线性插值有利于平衡增强程度与块边界连续性。
- 如后续取得原始脚本、原始图片与报告，按独立提交补齐并重新核对。

## MATLAB补充实验

找回了 `lab01_spatial/run_histogram_experiment.m` 以及 `lab03_motion_psf/matlab/` 下的两份运动模糊PSF MATLAB脚本；需要安装MATLAB及相关工具箱，并准备实验输入。本次未重跑MATLAB，历史报告中63组PSF对比结果不能当作本次复现实测。运动模糊PSF比较与逆滤波/维纳反卷积是不同任务，后者不作为已完成实验。

## 来源与署名

整理依据为本人课程实验报告（2026年9—11月）：《LHE与CLAHE原理实现及性能对比实验报告》《数字图像处理实验一》《实验二：图像增强的频域方法》。为保护学生信息和课程原始图片版权，仓库暂不上传包含个人信息的完整成绩/报告PDF及未核实授权的原图。
