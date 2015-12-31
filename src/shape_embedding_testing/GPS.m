% posToAnchorDist (n by m): distance from sensor points to anchor points (per row)
% posToSensorDist (n by n): distance from sensor points to sensor points (per row)
% anchorLoc (m by d): anchor point location per row
% estLoc (n by d): estimation point location
% x0: initial location of sensor points

function estLoc = GPS(posToAnchorDist, posToSensorDist, anchorLoc, params)
n = size(posToAnchorDist, 1);
m = size(anchorLoc, 1);
d = size(anchorLoc, 2);

if isfield(params, 'x0')
    x0 = params.x0;
else
    w = 1 ./ posToAnchorDist;
    w = bsxfun(@rdivide, w, sum(w, 2));
    x0 = w * anchorLoc;
    x0 = rand(n, d);
    %     [~, closestPoint] = min(posToAnchorDist, [], 2);
    %     x0 = anchorLoc(closestPoint, :)+1e-3 * rand(n,d);
end
if isfield(params, 'weightForAnchors')
    weightForAnchors = params.weightForAnchors;
else
    if strcmp(params.type, 'sammon')
        weightForAnchors = 1 ./ (posToAnchorDist + 1e-6);
    else
        weightForAnchors = ones(n, m);
    end
end

if isfield(params, 'weightForSensor')
    weightForSensor = params.weightForSensor;
else
    if strcmp(params.type, 'sammon')
        weightForSensor = 1 ./ (posToSensorDist + 1e-6);
        weightForSensor = weightForSensor .* (ones(size(weightForSensor)) - eye(size(weightForSensor)));
    else
        weightForSensor = ones(n, n);
    end
end

weightForAnchors = weightForAnchors / (sum(sum(weightForAnchors)) + sum(sum(weightForSensor)));
weightForSensor = weightForSensor / (sum(sum(weightForAnchors)) + sum(sum(weightForSensor)));

fprintf('Estimating positions for query points...\n');
obj = @(x)leastSquare(x, posToAnchorDist, posToSensorDist, anchorLoc, weightForAnchors, weightForSensor);

x0 = x0(:);
options.numDiff = 0;
estLoc = minFunc(obj, x0, options);
estLoc = reshape(estLoc, [n, d]);

end

function [f, g] = leastSquare(x, posToAnchorDist, posToSensorDist, anchorLoc, weightForAnchors, weightForSensor)
% 'call'
n = size(posToAnchorDist, 1);
d = size(anchorLoc, 2);
x = reshape(x, [n, d]);
f = 0;
g = zeros(n, d);
for i = 1:n
    tmp = bsxfun(@minus, x(i, :), anchorLoc);
    posToAnchorDistEst = sqrt(sum(tmp.^2, 2));
    res = posToAnchorDistEst' - posToAnchorDist(i, :);
    f = f + 0.5 * weightForAnchors(i, :) * (res.^2)';    
    g(i, :) = g(i, :) + (weightForAnchors(i, :) .* res ./ (posToAnchorDistEst + 1e-12)') * tmp;        
end
for i = 1:n
    tmp = bsxfun(@minus, x(i, :), x);
    posToSensorDistEst = sqrt(sum(tmp.^2, 2));
    res = posToSensorDistEst' - posToSensorDist(i, :);
    f = f + 0.5 * weightForSensor(i, :) * (res.^2)';
    g(i, :) = g(i, :) + (weightForSensor(i, :) .* res ./ (posToSensorDistEst + 1e-10)') * tmp;
end
g = g(:);
end
