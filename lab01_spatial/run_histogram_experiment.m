%% 实验一：灰度直方图、均衡化与规定化（手写算法 + MATLAB 官方结果对比）
% 对应：武汉理工大学《数字图像处理》实验一第一部分
% 注意：直方图统计、CDF、均衡化、规定化均自行实现。
%       histeq 仅在对照组中调用，不参与任何手写算法。
% MATLAB 官方函数名是 histeq，不是指导书笔误的 imhisteq。
% 要求：MATLAB + Image Processing Toolbox（仅对照组需要该 Toolbox）。

clear; clc; close all;
projectDir = fileparts(mfilename('fullpath'));
inputName = fullfile(projectDir, '..', 'inputs', 'face_512_gray.png');
outDir = fullfile(projectDir, 'results_matlab');
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% 1. 读入已经预处理为 512×512 的真实人脸灰度图
I = imread(inputName);                    % 仅文件读取，不使用现成增强算法
if ndims(I) == 3
    % 若输入是 RGB，人工按标准亮度系数转换；不调用 rgb2gray
    I = uint8(round(0.299*double(I(:,:,1)) ...
                  + 0.587*double(I(:,:,2)) ...
                  + 0.114*double(I(:,:,3))));
end
if ~isa(I, 'uint8'), error('本实验应输入 8 位灰度图（uint8）。'); end
if ~isequal(size(I), [512, 512])
    error('输入图片必须先调整为 512×512：当前为 %d×%d。', size(I,1), size(I,2));
end

%% 2. 手写直方图计算（不用 imhist / histogram 做统计）
hOriginal = manualHistogram256(I);
assert(sum(hOriginal) == numel(I), '直方图计数校验未通过！');

%% 3. 手写直方图均衡化：统计 -> CDF -> LUT -> 灰度映射
[JManual, hManual, mapEq, cdfSource] = manualEqualization(I);

%% 4. 手写直方图规定化：自定义双峰目标分布
levels = (0:255)';
targetRaw = 0.56*exp(-0.5*((levels-67)/18).^2) ...
          + 0.44*exp(-0.5*((levels-191)/24).^2);
targetProbability = targetRaw / sum(targetRaw);
[JSpecified, hSpecified, mapSpec] = manualSpecification(I, targetProbability);

%% 5. 仅在“对照组”使用 MATLAB 官方函数
% histeq(I) 默认按 64 个区间处理，与手写的 256 级实现并非同一设置。
% 所以将 histeq(I,256) 作为主要对照，同时展示默认 histeq(I)。
if exist('histeq', 'file') ~= 2
    error(['未检测到 Image Processing Toolbox 中的 histeq。', ...
           '手写算法本身不需要该工具箱，官方对照请在安装工具箱后运行。']);
end
JOfficial256 = histeq(I, 256);     % 对照组：256 bins
JOfficial64  = histeq(I);          % 对照组：默认 64 bins
JOfficialSpecified = histeq(I, targetProbability * numel(I));  % 对照组：指定目标

%% 6. 给出可核查的数值指标（不要求手写与官方逐像素相同）
hOfficial = manualHistogram256(JOfficial256);
hOfficialSpecified = manualHistogram256(JOfficialSpecified);
meanAbsoluteDifference = mean(abs(double(JManual(:)) - double(JOfficial256(:))));
rootMeanSquareDifference = sqrt(mean((double(JManual(:)) - double(JOfficial256(:))).^2));
maeSpecification = mean(abs(double(JSpecified(:)) - double(JOfficialSpecified(:))));

fprintf('\n========= 直方图实验结果 =========\n');
fprintf('输入尺寸：%d × %d\n', size(I,1), size(I,2));
fprintf('原始图像：均值 %.4f，标准差 %.4f\n', mean(double(I(:))), std(double(I(:))));
fprintf('手写均衡化：均值 %.4f，标准差 %.4f\n', mean(double(JManual(:))), std(double(JManual(:))));
fprintf('官方 histeq(I,256)：均值 %.4f，标准差 %.4f\n', ...
    mean(double(JOfficial256(:))), std(double(JOfficial256(:))));
fprintf('手写 vs 官方256级均衡化：MAE = %.6f, RMSE = %.6f\n', ...
    meanAbsoluteDifference, rootMeanSquareDifference);
fprintf('手写 vs 官方规定化：MAE = %.6f\n', maeSpecification);
fprintf('解释：不同离散灰度映射的优化策略不同，因此 MAE 不一定为 0。\n');

%% 7. 第一张图：原图、手写均衡化、histeq 256 及相应手写直方图
f1 = figure('Color','w','Position',[80,80,1200,680]);
subplot(2,3,1); showGray(I); title('Original image 512x512');
subplot(2,3,2); showGray(JManual); title('Handwritten equalization');
subplot(2,3,3); showGray(JOfficial256); title('MATLAB histeq(I,256)');
subplot(2,3,4); plotHist(hOriginal); title('Original histogram');
subplot(2,3,5); plotHist(hManual); title('Handwritten histogram');
subplot(2,3,6); plotHist(hOfficial); title('MATLAB histeq histogram');
saveas(f1,fullfile(outDir,'figure1_equalization_compare.png'));

%% 8. 第二张图：规定化与官方规定化，以及期望目标分布
f2 = figure('Color','w','Position',[100,100,1200,680]);
subplot(2,3,1); showGray(I); title('Original image');
subplot(2,3,2); showGray(JSpecified); title('Handwritten specification');
subplot(2,3,3); showGray(JOfficialSpecified); title('MATLAB histeq(I,target)');
subplot(2,3,4); plot(levels,targetProbability,'LineWidth',1.3); xlim([0,255]); ...
    xlabel('Gray level'); ylabel('Probability'); title('Bimodal target (probability)');
subplot(2,3,5); plotHist(hSpecified); title('Handwritten specified histogram');
subplot(2,3,6); plotHist(hOfficialSpecified); title('MATLAB specified histogram');
saveas(f2,fullfile(outDir,'figure2_specification_compare.png'));

%% 9. 第三张图：手写映射函数 + 官方不同灰度区间的差别
f3 = figure('Color','w','Position',[120,120,1080,420]);
subplot(1,3,1); plot(levels,double(mapEq),'LineWidth',1.1); ...
    xlabel('Input gray level'); ylabel('Output gray level'); title('Equalization LUT'); ...
    xlim([0,255]); ylim([0,255]); grid on;
subplot(1,3,2); plot(levels,double(mapSpec),'LineWidth',1.1); ...
    xlabel('Input gray level'); ylabel('Output gray level'); title('Specification LUT'); ...
    xlim([0,255]); ylim([0,255]); grid on;
subplot(1,3,3); showGray(JOfficial64); title('Default histeq(I): 64 bins');
saveas(f3,fullfile(outDir,'figure3_mapping_and_default.png'));

%% 10. 导出实际实验结果
imwrite(JManual,fullfile(outDir,'manual_equalized.png'));
imwrite(JSpecified,fullfile(outDir,'manual_specified.png'));
imwrite(JOfficial256,fullfile(outDir,'matlab_histeq256.png'));
imwrite(JOfficialSpecified,fullfile(outDir,'matlab_histeq_target.png'));
metricFile = fopen(fullfile(outDir,'result_metrics.txt'),'w');
assert(metricFile>0,'写入指标文件失败');
fprintf(metricFile,'Input 512x512 uint8 gray\n');
fprintf(metricFile,'Manual vs histeq(I,256) MAE = %.6f\n',meanAbsoluteDifference);
fprintf(metricFile,'Manual vs histeq(I,256) RMSE = %.6f\n',rootMeanSquareDifference);
fprintf(metricFile,'Manual specification vs histeq(I,target) MAE = %.6f\n',maeSpecification);
fclose(metricFile);

fprintf('已输出 3 张对照图、4 张结果图和 result_metrics.txt 到 %s\n',outDir);

%% 以下为“手写核心算法”，严禁以 histeq/imhist/imhistmatch 替代。
function h = manualHistogram256(A)
    % 统计灰度 0...255 出现的次数，h(1) 代表灰度 0
    h = zeros(256,1);
    v = A(:);
    for i = 1:numel(v)
        k = double(v(i)) + 1;   % MATLAB 下标从 1 开始
        h(k) = h(k) + 1;
    end
end

function [J,hJ,lut,cdf] = manualEqualization(I)
    h = manualHistogram256(I);
    p = h / double(numel(I));
    cdf = zeros(256,1);
    running = 0;
    for k = 1:256
        running = running + p(k);
        cdf(k) = running;
    end
    % 选用首个非零 CDF 的归一化版本：把最暗有效灰度映射到 0。
    first = find(h>0,1,'first');
    cdfMin = cdf(first);
    if cdfMin >= 1-1e-12
        lut = uint8((0:255)'); % 常值图像无需增强
    else
        lut = uint8(round(255*max(0,min(1,(cdf-cdfMin)/(1-cdfMin)))));
    end
    J = reshape(lut(double(I(:))+1),size(I));
    hJ = manualHistogram256(J);
end

function [J,hJ,lut] = manualSpecification(I,targetP)
    if numel(targetP) ~= 256 || any(targetP(:)<0) || sum(targetP(:))<=0
        error('目标直方图必须是 256 维非负非零向量！');
    end
    h = manualHistogram256(I);
    p = h / double(numel(I));
    q = targetP(:)/sum(targetP(:));
    F = zeros(256,1); G = zeros(256,1);
    runningF = 0; runningG = 0;
    for k = 1:256
        runningF = runningF + p(k);
        runningG = runningG + q(k);
        F(k) = runningF;
        G(k) = runningG;
    end
    % 逐级查找目标 CDF 中距离源 CDF 最近的灰度等级。
    lut = zeros(256,1,'uint8');
    for k = 1:256
        [~,pos] = min(abs(G-F(k)));
        lut(k) = uint8(pos-1);
    end
    J = reshape(lut(double(I(:))+1),size(I));
    hJ = manualHistogram256(J);
end

function showGray(A)
    % 用 MATLAB 基础图形函数显示，不调用 imshow 等图像工具箱算法
    imagesc(double(A),[0,255]); colormap(gray(256)); axis image off;
end

function plotHist(h)
    plot(0:255,h,'LineWidth',1.0); xlim([0,255]);
    xlabel('Gray level'); ylabel('Pixel count');
end
