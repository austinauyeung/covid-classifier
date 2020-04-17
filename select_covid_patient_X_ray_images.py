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
x_ray_view = ["AP", "AP semi erect", "AP Supine", "Coronal", "PA"] # View of X-Ray

metadata = os.getcwd()+'/metadata.csv' # Meta info
imageDir = os.getcwd()+'/images' # Directory of images
outputDir = os.getcwd()+'/output' # Output directory to store selected images
try:
	os.mkdir(outputDir)
except:
	print('Output directory already exists.')
metadata_csv = pd.read_csv(metadata)

# loop over the rows of the COVID-19 data frame
for (i, row) in metadata_csv.iterrows():
	if row["finding"] != virus or (row["view"] not in x_ray_view):
		continue

	try:
		filename = row["filename"].split(os.path.sep)[-1]
		inputPath = os.path.sep.join([imageDir, filename])
		outputPath = os.path.sep.join([outputDir, filename])
		shutil.move(inputPath, outputPath)
	except:
		print(filename+' not found, may have been moved already.')

