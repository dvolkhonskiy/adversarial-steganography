clc; clear all;

pkg load image;

files = dir(fullfile('/', 'home', 'dvolkhonskiy', 'datasets', 'stego_celeb', 'hugo_test', '*.png'));

% set payload
payload = 0.4;

% set params
params.gamma = 1;
params.sigma = 1;

for file = files'

    fprintf(file.name)
    fprintf('\n')

    image = imread(fullfile('/', 'home', 'dvolkhonskiy', 'datasets', 'stego_celeb', 'hugo_test', file.name));

    fprintf('Embedding using matlab code\n');

    %% Run embedding simulation
    [stego, distortion] = HUGO_like(image(:, :, 1), payload, params);
    image(:, :, 1) = stego;

    %[stego, distortion] = HUGO_like(image(:, :, 2), payload, params);
    %image(:, :, 2) = stego;

    %[stego, distortion] = HUGO_like(image(:, :, 3), payload, params);
    %image(:, :, 3) = stego;


    imwrite(image, strcat('/home/dvolkhonskiy/datasets/stego_celeb/hugo_test/stego_', file.name), 'png')

    fprintf(' - DONE\n');
end

