clc
clear
close all

data = readtable('/Users/oviyat/Desktop/CX/data/Data_Entry_2017.csv');
classes = data(:,2);

lenclasses = size(classes);
lenclasses = lenclasses(1);

classes = classes.Var2;
cardio = zeros(lenclasses,1); %cardiomegaly only array
edema = zeros(lenclasses,1); %edema only array
healthy = zeros(lenclasses,1); %healthy array


for i = 1:lenclasses
    diseases = strsplit(cell2mat(classes(i,1)),'|');
    lend = length(diseases);
    for ind = 1:lend
        both = 0;
        if isequal(cell2mat(diseases(ind)),'Cardiomegaly')
             if edema(i) == 1
                edema(i) = 0;
                cardio(i) = 0;
            else
                cardio(i) = 1;
            end
        elseif isequal(cell2mat(diseases(ind)),'Edema')
            if cardio(i) == 1
                edema(i) = 0;
                cardio(i) = 0;
            else
                edema(i) = 1;
            end
        elseif isequal(cell2mat(diseases(ind)),'No Finding')
            healthy(i) = 1;
        else
        end
    end
end

totcardio = sum(cardio);
totedema = sum(edema);
tothealthy = sum(healthy);

filenames = data(:,1);

filenames = filenames.Var1;

[rc,~] = find(cardio); %index values for cardiomegaly
[re,~] = find(edema); %index values for edema

%mkdir '/Users/oviyat/Desktop/CX/cardiomegaly'
%mkdir '/Users/oviyat/Desktop/CX/edema'

% for i = 1:length(rc)
%     car = filenames{rc(i),1};
%     filename = ['/Users/oviyat/Desktop/CX/theimages/',car];
%     A = imread(filename);
%     imwrite(A,['/Users/oviyat/Desktop/CX/cardiomegaly/',car]);
% end

for i = 1:length(re)
    ed = filenames{re(i),1};
    filename = ['/Users/oviyat/Desktop/CX/theimages/',ed];
    A = imread(filename);
    imwrite(A,['/Users/oviyat/Desktop/CX/edema/',ed]);
end


