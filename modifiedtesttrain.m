%this code was written by Oviya
clc
clear
close all

% fid=fopen('train_val_list.txt');
% tline = fgetl(fid);
% tlines = cell(0,1);
% while ischar(tline)
%     tlines{end+1,1} = tline;
%     tline = fgetl(fid);
% end
% fclose(fid);

listingtest = dir('/Users/oviyat/Desktop/CX/test');
listingtrain = dir('/Users/oviyat/Desktop/CX/train');

for i =4: length(listingtest)
    imgname = listingtest(i).name;
    fid = fopen('train_val_list.txt', 'a');
    fprintf(fid, [imgname,'\n']);
    fclose(fid);
end

for i = 4: length(listingtrain)
    imgname = listingtrain(i).name;
    fid = fopen('test_list.txt', 'a');
    fprintf(fid, [imgname,'\n']);
    fclose(fid);
end

 