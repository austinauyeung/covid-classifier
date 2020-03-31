clc
clear
close all

source = '/Users/oviyat/Desktop/CX';
data = readtable([source '/data/Data_Entry_2017.csv']);

classes = data.Var2;
lenclasses = size(classes,1);

cardio = zeros(lenclasses,1); %cardiomegaly only array
edema = zeros(lenclasses,1); %edema only array
healthy = zeros(lenclasses,1); %healthy array

for i = 1:lenclasses
    disp(['Image ' num2str(i) ' out of ' num2str(lenclasses)]);
    diseases = strsplit(cell2mat(classes(i)),'|');
    lend = length(diseases);
    if lend == 1
        if isequal(cell2mat(diseases(1)),'Cardiomegaly')
            cardio(i) = 1;
        elseif isequal(cell2mat(diseases(1)),'Edema')
            edema(i) = 1;
        elseif isequal(cell2mat(diseases(1)),'No Finding')
            healthy(i) = 1;
        else
        end
    end
end

totcardio = sum(cardio);
totedema = sum(edema);
tothealthy = sum(healthy);

filenames = data.Var1;

[rc,~] = find(cardio); %index values for cardiomegaly
[re,~] = find(edema); %index values for edema

% mkdir '/Users/oviyat/Desktop/CX/cardiomegaly2'
% mkdir '/Users/oviyat/Desktop/CX/edema2'

for i = 1:length(rc)
    car = filenames{rc(i),1};
    filename = [source '/theimages/' car];
    dest = [source '/cardiomegaly2/' car];
    movefile(filename,dest);
end

for i = 1:length(re)
    ed = filenames{re(i),1};
    filename = [source '/theimages/' ed];
    dest = [source '/edema2/' ed];
    movefile(filename,dest);
end


