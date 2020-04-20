clc
clear
close all

source = [pwd '/'];

listings = dir([source,'images_processed_2/*.png']);
c = [];
h = [];
p = [];
u = [];
clast = {};
hlast = {};
plast = {};

mkdir('covid_processed');
mkdir('healthy_processed');
mkdir('pne_processed');
mkdir('unique_processed');

% create square image by either cropping or padding with zeros
crop = 1;

%split into each class
cdir = [source,'covid_processed/'];
hdir = [source,'healthy_processed/'];
pdir = [source,'pne_processed/'];
for i = 1:length(listings)
    
    filenames = listings(i).name;
    fileinfo = strsplit(filenames,'_');
    
    if isequal(fileinfo{1},'CO')
        c = [c;str2num(fileinfo{2})];
        clast{end+1} = fileinfo{3};
    elseif isequal(fileinfo{1},'No')
        h = [h;str2num(fileinfo{2})];
        hlast{end+1} = fileinfo{3};
    elseif isequal(fileinfo{1},'Pn')
        p = [p;str2num(fileinfo{2})];
        plast{end+1} = fileinfo{3};
    else
    end
    
end

%find unique patient
[cpatient,cindx,~] = unique(c);
[ppatient,pindx,~] = unique(p);
[hpatient,hindx,~] = unique(h);

for i = 1:length(cindx)
    covid = strcat('CO_',num2str(cpatient(i)),'_',clast{1,cindx(i)});
    filename = strcat(source,'images_processed_2/',covid);
    A = imread(filename);
    
    sizediff = abs(size(A,1)-size(A,2));
    side2 = floor(sizediff/2);
    side1 = sizediff-side2;
    if crop
        if sizediff>0 && size(A,1)>size(A,2) % tall
            A = A(side1+1:end-side2,:,:);
        elseif sizediff>0 && size(A,1)<size(A,2) % wide
            A = A(:,side1+1:end-side2,:);
        end
    else
        if sizediff>0 && size(A,1)>size(A,2) % tall
            pad1 = zeros(size(A,1),side1,size(A,3));
            pad2 = zeros(size(A,1),side2,size(A,3));
            A = cat(2,pad1,A,pad2);
        elseif sizediff>0 && size(A,1)<size(A,2) % wide
            pad1 = zeros(side1,size(A,2),size(A,3));
            pad2 = zeros(side2,size(A,2),size(A,3));
            A = cat(1,pad1,A,pad2);
        end
    end
    
    dest = [cdir covid];
    imwrite(A,dest);
end

for i = 1:length(hindx)
    healthy = strcat('No_',num2str(hpatient(i)),'_',hlast{1,hindx(i)});
    filename = strcat(source,'images_processed_2/',healthy);
    A = imread(filename);

    sizediff = abs(size(A,1)-size(A,2));
    side2 = floor(sizediff/2);
    side1 = sizediff-side2;
    if crop
        if sizediff>0 && size(A,1)>size(A,2) % tall
            A = A(side1+1:end-side2,:,:);
        elseif sizediff>0 && size(A,1)<size(A,2) % wide
            A = A(:,side1+1:end-side2,:);
        end
    else
        if sizediff>0 && size(A,1)>size(A,2) % tall
            pad1 = zeros(size(A,1),side1,size(A,3));
            pad2 = zeros(size(A,1),side2,size(A,3));
            A = cat(2,pad1,A,pad2);
        elseif sizediff>0 && size(A,1)<size(A,2) % wide
            pad1 = zeros(side1,size(A,2),size(A,3));
            pad2 = zeros(side2,size(A,2),size(A,3));
            A = cat(1,pad1,A,pad2);
        end
    end
    
    dest = [hdir healthy];
    imwrite(A,dest);
end

for i = 1:length(pindx)
    pneumonia = strcat('Pn_',num2str(ppatient(i)),'_',plast{1,pindx(i)});
    filename = strcat(source, 'images_processed_2/', pneumonia);
    A = imread(filename);

    sizediff = abs(size(A,1)-size(A,2));
    side2 = floor(sizediff/2);
    side1 = sizediff-side2;
    if crop
        if sizediff>0 && size(A,1)>size(A,2) % tall
            A = A(side1+1:end-side2,:,:);
        elseif sizediff>0 && size(A,1)<size(A,2) % wide
            A = A(:,side1+1:end-side2,:);
        end
    else
        if sizediff>0 && size(A,1)>size(A,2) % tall
            pad1 = zeros(size(A,1),side1,size(A,3));
            pad2 = zeros(size(A,1),side2,size(A,3));
            A = cat(2,pad1,A,pad2);
        elseif sizediff>0 && size(A,1)<size(A,2) % wide
            pad1 = zeros(side1,size(A,2),size(A,3));
            pad2 = zeros(side2,size(A,2),size(A,3));
            A = cat(1,pad1,A,pad2);
        end
    end
    
    dest = [pdir pneumonia];
    imwrite(A,dest);
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