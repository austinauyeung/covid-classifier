'''
This code finds all images of patients of a specified VIRUS and X-Ray view and stores selected image to an OUTPUT directory
+ It uses metadata.csv for searching and retrieving images name
+ Using ./images folder it selects the retrieved images and copies them in output folder
Code can be modified for any combination of selection of images
'''

import pandas as pd
import argparse
import shutil
import os

# Selecting all combination of 'COVID-19' patients with 'PA' X-Ray view
virus = "COVID-19" # Virus to look for
x_ray_view = "AP" # View of X-Ray #Oviya: we want: AP, AP semi erect, AP Supine, Coronal, PA

metadata = '/Users/oviyat/Desktop/CX/covid-chestxray-dataset-master/covid-chestxray-dataset-master/metadata.csv' # Meta info
imageDir = '/Users/oviyat/Desktop/CX/covid-chestxray-dataset-master/covid-chestxray-dataset-master/images'
#imageDir = "./images" # Directory of images
outputDir = '/Users/oviyat/Desktop/CX/covid-chestxray-dataset-master/covid-chestxray-dataset-master/output' # Output directory to store selected images

metadata_csv = pd.read_csv(metadata)

# loop over the rows of the COVID-19 data frame
for (i, row) in metadata_csv.iterrows():
	if row["finding"] != virus or row["view"] != x_ray_view:
		continue

	filename = row["filename"].split(os.path.sep)[-1]
	imagepath = os.path.sep.join([imageDir, filename]) #oviya line
	outputPath = os.path.sep.join([outputDir, filename])
	#shutil.copy2(imageDir, outputPath)
	shutil.move(imagepath, outputPath) #oviya line

