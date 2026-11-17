% TEST_MOTION_PSF  对比自己实现的运动模糊 PSF 与 MATLAB fspecial。
% 将本脚本与 my_motion_psf.m 放在同一个文件夹中运行。
clear; clc; close all;
assert(exist('fspecial', 'file') ~= 0, ...
    '未找到 fspecial：需要安装 Image Processing Toolbox。');

% 多个运动长度与角度组合
LENs = [3, 5, 9, 15, 20, 25, 31];
THETAs = [0, 15, 30, 45, 60, 90, 120, 135, 179];
N = numel(LENs) * numel(THETAs);
results = zeros(N, 4);  % [LEN, THETA, MaxError, SumCustom]
maxErr = 0;
idx = 0;
for len = LENs
    for theta = THETAs
        idx = idx + 1;
        A = my_motion_psf(len, theta);
        B = fspecial('motion', len, theta);
        assert(isequal(size(A), size(B)), ...
            '尺寸不一致: LEN=%g, THETA=%g', len, theta);
        err = max(abs(A(:) - B(:)));
        results(idx, :) = [len, theta, err, sum(A(:))];
        maxErr = max(maxErr, err);
        fprintf('LEN=%2g, THETA=%3g : 最大绝对误差 = %.4e\n', ...
            len, theta, err);
    end
end
fprintf('\n共 %d 组，最大绝对误差 = %.16e\n', N, maxErr);
if maxErr < 1e-12
    fprintf('测试结论：所有测试通过，与 fspecial 达到数值精度一致。\n');
else
    fprintf('测试结论：存在差异，请检查对应参数。\n');
end

% 可视化对比（LEN=20、THETA=45）
A = my_motion_psf(20, 45);
B = fspecial('motion', 20, 45);
figure('Color','w', 'Name','运动模糊 PSF 对比');
subplot(1, 3, 1); imagesc(A); axis image; colorbar; title('自己编写');
subplot(1, 3, 2); imagesc(B); axis image; colorbar; title('MATLAB fspecial');
subplot(1, 3, 3); imagesc(abs(A-B)); axis image; colorbar; title('逐元素绝对误差');
saveas(gcf, 'motion_psf_comparison.png');

% 保存测试数据，方便复制到实验报告
fid = fopen('motion_psf_results.csv', 'w');
fprintf(fid, 'LEN,THETA,MaxAbsError,SumCustom\n');
fprintf(fid, '%g,%g,%.16e,%.16f\n', results.');
fclose(fid);
fprintf('已保存：motion_psf_comparison.png 与 motion_psf_results.csv\n');
