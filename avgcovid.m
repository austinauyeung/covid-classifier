
clc
clear
close all

files=dir('/Users/oviyat/Desktop/CX/covid_large');

total = zeros(224,224,3);
total=double(total);

for k=3:length(files)
    covid=files(k).name;
    filename = ['/Users/oviyat/Desktop/CX/covid_large/',covid];
    A = imread(filename);
    A = imresize(A,[224 224]);
    A = double(A);
    total = total+A;
end

covidavg = total/(length(files)-2);
covidavg = uint8(covidavg);
imshow(covidavg);
imwrite(covidavg,'/Users/oviyat/Desktop/CX/covidavg.jpeg');