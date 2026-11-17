function H = my_motion_psf(LEN, THETA)
%MY_MOTION_PSF 运动模糊点扩散函数（独立实现，不调用 fspecial）
%   H = MY_MOTION_PSF(LEN, THETA)
%   LEN   : 运动长度，正的有限实数标量，单位为像素
%   THETA : 运动方向角，单位为度，逆时针为正
%
%   算法：构造以核中心为起点的半平面坐标网格，计算像素到有限
%   运动线段的距离，以 max(1-distance,0) 赋权，利用中心对称性
%   展开成完整核，最后归一化。

    if nargin < 1 || isempty(LEN)
        LEN = 9;
    end
    if nargin < 2 || isempty(THETA)
        THETA = 0;
    end

    validateattributes(LEN, {'numeric'}, ...
        {'scalar','real','finite','positive'}, mfilename, 'LEN', 1);
    validateattributes(THETA, {'numeric'}, ...
        {'scalar','real','finite'}, mfilename, 'THETA', 2);

    len = max(1, double(LEN));
    half = (len - 1) / 2;

    % 运动模糊方向以 180 度为周期。
    phi = mod(double(THETA), 180) / 180 * pi;
    cosphi = cos(phi);
    sinphi = sin(phi);
    xsign = sign(cosphi);

    lineWidth = 1;

    % 依据运动方向确定半核的离散支持范围。
    sx = fix(half * cosphi + lineWidth * xsign - len * eps);
    sy = fix(half * sinphi + lineWidth - len * eps);
    [x, y] = meshgrid(0:xsign:sx, 0:sy);

    % 点到无限直线的垂直距离。
    dist2line = y .* cosphi - x .* sinphi;
    radius = sqrt(x .* x + y .* y);

    % 对有限运动线段端点附近的像素，改用到端点的距离。
    lastPixel = find((radius >= half) & (abs(dist2line) <= lineWidth));
    if ~isempty(lastPixel)
        alongToEnd = half - abs( ...
            (x(lastPixel) + dist2line(lastPixel) .* sinphi) ./ cosphi);
        dist2line(lastPixel) = sqrt( ...
            dist2line(lastPixel).^2 + alongToEnd.^2);
    end

    % 距离越近权重越大，线宽之外的权重置零。
    dist2line = lineWidth + eps - abs(dist2line);
    dist2line(dist2line < 0) = 0;

    % 将半核按中心对称性展开为完整二维核。
    H = rot90(dist2line, 2);
    H(end + (1:size(dist2line, 1)) - 1, ...
      end + (1:size(dist2line, 2)) - 1) = dist2line;

    % 归一化，保留机器精度补偿项。
    H = H ./ (sum(H(:)) + eps * len * len);

    % MATLAB 图像矩阵的行方向向下，修正角度对应的矩阵方向。
    if cosphi > 0
        H = flipud(H);
    end
end
