# 文件来源与复原记录 / Provenance

| 文件 | 来源报告 | 恢复处理 | 验证状态 |
|---|---|---|---|
| `lab00_local_hist/lhe_clahe_lab.py` | 《LHE与CLAHE原理实现及性能对比实验报告》8–9页代码附录 | 从PDF提取；`&gt;`等转义符还原；输出从报告使用的临时路径改为脚本同级 `results/` | 已重运行并生成指标、图片 |
| `lab01_spatial/spatial_lab.py` | 《数字图像处理实验一_黄博远_数学JD2401》11–20页附录 | 保留核心公式函数；报告PDF分页导致入口代码行断裂，故按报告实验流程重构 `run()`，新增`--input`；没恢复原始人脸图片 | 核心函数编译通过，替代图片运行通过；原图结果未复现 |
| `lab02_frequency/frequency_lab.py` | 《实验二_图像增强的频域方法_黄博远_正式实验报告》12–20页附录F | 从PDF恢复代码；当 `input_images/` 四张原图缺失时跳过对应挑战，不影响固定种子模拟实验 | 自动非交互实验运行通过；原图周期陷波任务未复现 |

## 历史结果与当前重运行数据的边界

历史报告曾记录原始人像和周期干扰四图的部分数值。本仓库未把这些指标包装成重新验证的结果。README中的可复现指标仅对应本次当前代码实际运行的数据与参数。`skimage.data.camera()` 是用于环境测试的替代图片，**不是**原始人脸测试集。

## 隐私与第三方资源

- 不公开报告中的学生编号、完整课程评分材料和未验证许可的附件图片。
- 标准样例 `skimage.data.camera` 与 Matplotlib 示例照片由对应库分发；本仓库只调用库内样例，不复制其二进制文件。
- `face_input.jpg` 及四张周期噪声附件图需由使用者自行放入指定目录。不要将未知来源图片直接重新分发。

## MATLAB standalone sources

- `lab01_spatial/run_histogram_experiment.m` comes from the recovered standalone MATLAB script. Its input portrait file remains missing.
- `lab03_motion_psf/matlab/my_motion_psf.m` and `test_motion_psf.m` come from recovered standalone `.m` files and historical report. Not executed during this GitHub packaging.
