clc
clear
close all

source = '/Users/oviyat/Desktop/CX/';

listings = dir([source,'images_processed_2']);


for i = 4:length(listings)
    image = listings(i).name;
    filename = strcat(source,'images_processed_2/',image);
    A = imread(filename);
    A = A(126:end-125,:,3);
    imwrite(A,[source,'cropped_processed/',image]);
end