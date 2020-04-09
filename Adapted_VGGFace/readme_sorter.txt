This file (sorter.py) can be used to sort NIH dataset.

First dump all NIH images into one directory called images and create the following file structure:

sorter.py
Data_Entry_2017.csv
data
 |_____images
            |_____00000001_000.png
            |_____00000001_001.png
              .
              .
              .
              
Now run sorter to create test.txt and train.txt files containing full paths to Cardiomegaly and Normal (No Finding) images. Next use the following python function to create train and test dictionaries. Note that sorter can easily be modified to sort and return other diseases as well (No need to modify get_dataset).

def get_dataset(filename):
    dataset = defaultdict(list)
    class_name = None
    with open(filename, 'r') as datafile:
        lines = datafile.readlines()
        for line in lines:
            if '#' in line:
                class_name = line[1:-1]
            else:
                dataset[class_name].append(line[:-1])
    return dataset
