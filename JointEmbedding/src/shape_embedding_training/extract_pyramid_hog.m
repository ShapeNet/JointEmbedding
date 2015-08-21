function [hog_feature] = extract_pyramid_hog(image_filename, hog_image_size)
    I = imread(image_filename);
    I = imresize(I, [hog_image_size, hog_image_size]);
    h0 = hog(single(I));
    I = imresize(I, 0.5);
    h1 = hog(single(I));
    I = imresize(I, 0.5);
    h2 = hog(single(I));
    hog_feature = [h0(:); h1(:); h2(:)]';
end