clc
clear
close all

source = '/Users/oviyat/Desktop/CX/';

listings = dir([source,'new_images_processed_2']);
c = [];
h = [];
p = [];
u = [];
clast = {};
hlast = {};
plast = {};

%split into each class
for i = 4:length(listings)
    
    filenames = listings(i).name;
    fileinfo = strsplit(filenames,'_');
    
    if isequal(fileinfo{1},'CO')
        cdir = [source,'covid_processed/'];
        c = [c;str2num(fileinfo{2})];
        clast{end+1} = fileinfo{3};
    elseif isequal(fileinfo{1},'No')
        hdir = [source,'healthy_processed/'];
        h = [h;str2num(fileinfo{2})];
        hlast{end+1} = fileinfo{3};
    elseif isequal(fileinfo{1},'Pn')
        pdir = [source,'pne_processed/'];
        p = [p;str2num(fileinfo{2})];
        plast{end+1} = fileinfo{3};
    else
    end
    
end

%find unique patient
[cpatient,cindx] = unique(c);
[ppatient,pindx] = unique(p);
[hpatient,hindx] = unique(h);

for i = 1:length(cindx)
    covid = strcat('CO_',num2str(cpatient(i)),'_',clast{1,cindx(i)});
    filename = strcat(source,'new_images_processed_2/',covid);
    A = imread(filename);
    A = imresize(A,[224 224]);
    dest = [cdir covid];
    imwrite(A,dest);
    %movefile(filename,dest);
end

for i = 1:length(hindx)
    healthy = strcat('No_',num2str(hpatient(i)),'_',hlast{1,hindx(i)});
    filename = strcat(source,'new_images_processed_2/',healthy);
    A = imread(filename);
    A = imresize(A,[224 224]);
    dest = [hdir healthy];
    imwrite(A,dest);
    %movefile(filename,dest);
end

for i = 1:length(pindx)
    pneumonia = strcat('Pn_',num2str(ppatient(i)),'_',plast{1,pindx(i)});
    filename = strcat(source, 'new_images_processed_2/', pneumonia);
    A = imread(filename);
    %A = imresize(A,[224 224]);
    dest = [pdir pneumonia];
    imwrite(A,dest);
    %movefile(filename,dest);
end

[~,txt,~] = xlsread([source,'covidfigures.xlsx']);
for i = 1:length(txt)
    image = txt{i};
    sep = strsplit(image,'_');
    if isequal(sep{1},'CO')
        cdir = [source,'covid_processed/',image];
        if isfile(cdir)
            movefile(cdir,[source,'unique_processed/']);
        else
        end
    elseif isequal(sep{1},'No')
        hdir = [source,'healthy_processed/',image];
        if isfile(hdir)
            movefile(hdir,[source,'unique_processed/']);
        else
        end
    elseif isequal(sep{1},'Pn')
        pdir = [source,'pne_processed/',image];
        if isfile(pdir)
            movefile(pdir,[source,'unique_processed/']);
        else
        end
    else
    end
end