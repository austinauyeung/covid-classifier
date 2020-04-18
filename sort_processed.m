clc
clear
close all

source = '/Users/oviyat/Desktop/CX/';

listings = dir([source,'images_processed_2']);
c = [];
h = [];
p = [];
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
    filename = strcat(source,'images_processed_2/',covid);
    dest = [cdir covid];
    movefile(filename,dest);
end

for i = 1:length(hindx)
    healthy = strcat('No_',num2str(hpatient(i)),'_',hlast{1,hindx(i)});
    filename = strcat(source,'images_processed_2/',healthy);
    dest = [hdir healthy];
    movefile(filename,dest);
end

for i = 1:length(pindx)
    pneumonia = strcat('Pn_',num2str(ppatient(i)),'_',plast{1,pindx(i)});
    filename = strcat(source, 'images_processed_2/', pneumonia);
    dest = [pdir pneumonia];
    movefile(filename,dest);
end
