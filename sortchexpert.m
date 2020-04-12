tic
clc
clear
close all
%running code on chexpert pneumonia dataset to see if high accuracy values
%are due to optimation bias from using entire test set as validation set

source = '/Users/oviyat/Desktop/CX/';
data = readtable([source 'CheXpert-v1.0-small/train.csv']);

view = data.Frontal_Lateral;
pneu = data.Pneumonia;
cxpath = data.Path;
lendata = size(cxpath,1);
otherdisease = data(:,[6 7 8 9 10 11 12 14 15 16 17 18]);
otherdisease = table2array(otherdisease);
odlength = size(otherdisease(1,:));
odlength = odlength(2);

pneumonia = zeros(lendata,1); %pneumonia array
pneupaths = [];
is = [];

for i = 1:lendata
    %disp(['Image ' num2str(i) ' out of ' num2str(lenclasses)]);
    only = 1; % the person only has pneumonia
    if pneu(i) == 1 && isequal(cell2mat(view(i)),'Frontal')
        for int = 1:odlength
            if otherdisease(i,int)== 1
                only = 0;
            else
            end
        end
        if only ==1
            pneupaths = [pneupaths;cxpath(i)];
            is = [is; i];
        end
    end
end

totpneu = size(pneupaths);
totpneu = totpneu(1);
checkout = data(is,:);

for i = 1:totpneu
    pn = pneupaths{i,1};
    filename = ['/Users/oviyat/Desktop/CX/',pn];
    A = imread(filename);
    A = imresize(A,[224 224]);
    imwrite(A,['/Users/oviyat/Desktop/CX/pneumonia_chexpert/',num2str(i),'.png']);
end

toc
