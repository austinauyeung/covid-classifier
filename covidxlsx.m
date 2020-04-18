clc
clear
close all

source = '/Users/oviyat/Desktop/CX/';

[num,txt,raw] = xlsread([source,'covidfigures.xlsx']);

for i = 1:length(txt)
    image = txt{i};
    filename = strcat(source,'images_processed_2/',image);
    dest = [source,'unique_processed/',image];
    movefile(filename,dest);
end