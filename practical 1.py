% Read the image
image = imread('Red_Kitten_01.jpg');

% Display original image
figure;
imshow(image);
title('Original Image');

% Separate RGB planes
R = image(:,:,1); % Red plane
G = image(:,:,2); % Green plane
B = image(:,:,3); % Blue plane

% Display RGB planes
figure;
subplot(1,3,1), imshow(R), title('Red Plane');
subplot(1,3,2), imshow(G), title('Green Plane');
subplot(1,3,3), imshow(B), title('Blue Plane');

% Sampling (reduce resolution)
sampledImage = image(1:2:end, 1:2:end, :);

figure;
imshow(sampledImage);
title('Sampled Image');

% Quantization
numLevels = 16;
quantizedImage = floor(double(image) / (256 / numLevels)) * (256 / numLevels);

figure;
imshow(uint8(quantizedImage));
title('Quantized Image');

% False contouring (extreme quantization)
numLevelsExtreme = 4;
falseContourImage = floor(double(image) / (256 / numLevelsExtreme)) * (256 / numLevelsExtreme);

figure;
imshow(uint8(falseContourImage));
title('False Contour Image');
