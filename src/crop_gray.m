function [I, top, bottom, left, right] = crop_gray(I, bg_color, jitter, cropRatios)
[nr, nc] = size(I);
col_sum = sum(I == bg_color, 1) ~= nr;
row_sum = sum(I == bg_color, 2) ~= nc;

left = find(col_sum, 1, 'first');
if left == 0
    left = 1;
end
right = find(col_sum, 1, 'last');
if right == 0
    right = length(col_sum);
end
top = find(row_sum, 1, 'first');
if top == 0
    top = 1;
end
bottom = find(row_sum, 1, 'last');
if bottom == 0
    bottom = length(row_sum);
end

if jitter == 0
    I = I(top:bottom, left:right, :);
    % I = padarray(I, [3, 3], bg_color, 'both');
    return;
end

% ---- JITTER THE CROP -----
if nargin < 4
    cropRatios = ones(1,4)*0.05;
end

width = right - left + 1;
height = bottom - top + 1;

% crop
dx1 = width * cropRatios(3) * randn;
if abs(dx1) > 0.4*width
    dx1 = 0;
end
dx2 = -1 * width * cropRatios(4) * randn;
if abs(dx2) > 0.4*width
    dx2 = 0;
end
dy1 = height * cropRatios(1) * randn;
if abs(dy1) > 0.3*height
    dy1 = 0;
end
dy2 = height * cropRatios(2) * 0.1 - 1 * height * cropRatios(2) * abs(randn);
if abs(dy2) > 0.6*height
    dy2 = 0;
end

leftnew = max([1, left + dx1]);
leftnew = min([leftnew, nc]);
rightnew = max([1, right + dx2]);
rightnew = min([rightnew, nc]);
if leftnew > rightnew
    leftnew = left;
    rightnew = right;
end

topnew = max([1, top + dy1]);
topnew = min([topnew, nr]);
bottomnew = max([1, bottom + dy2]);
bottomnew = min([bottomnew, nr]);
if topnew > bottomnew
    topnew = top;
    bottomnew = bottom;
end

left = round(leftnew); right = round(rightnew);
top = round(topnew); bottom = round(bottomnew);
I = I(top:bottom, left:right, :);
