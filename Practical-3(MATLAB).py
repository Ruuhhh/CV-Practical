I = imread('Elephant.jpg');
if isempty(I)
    error('Image not found. Check the image path.');
end
[H, W, ~] = size(I);
newW = 400;
newH = round(newW * (H / W));
Ires = imresize(I, [newH newW]);
if size(Ires,3) == 3
    Igray = rgb2gray(Ires);
else
    Igray = Ires;
end
[features, vis] = extractHOGFeatures(Igray, 'CellSize', [8 8]);
fig = figure('Visible','off', 'Units','pixels', 'Position',[100 100 600 400]);
plot(vis);
axis off;
drawnow;
frame = getframe(gca);
visImgColor = frame.cdata;
close(fig);
visImgGray = rgb2gray(visImgColor);
visImgScaled = mat2gray(visImgGray);
figure;
subplot(1,2,1), imshow(Ires), title('Resized Original');
subplot(1,2,2), imshow(visImgScaled), title('HOG Visualization (captured)');
fprintf('Original image size: %d x %d x %d\n', size(I));
fprintf('Resized image size: %d x %d x %d\n', size(Ires));
fprintf('HOG feature vector length: %d\n', numel(features));
fprintf('Captured HOG image size: %d x %d x %d\n', size(visImgColor));
