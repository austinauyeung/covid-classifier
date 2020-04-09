This file can be used to sort NIH dataset.

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
              
Now run sorter to create test.txt and train.txt files containing full paths to Cardiomegaly and Normal (No Finding) images. Next use 
