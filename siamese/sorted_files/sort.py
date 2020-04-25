import argparse
from PIL import Image

from os import listdir, mkdir
from os.path import isfile, join, exists


parser = argparse.ArgumentParser(description="Sorts and preprocess dataset")
parser.add_argument("image_width", type=int)
parser.add_argument("image_height", type=int)
parser.add_argument("--co", default="covid", help="COVID image directory")
parser.add_argument("--no", default="healthy", help="Healthy image directory")
parser.add_argument("--pn", default="pneumonia", help="Pneumonia image directory")
args = parser.parse_args()

if not exists(args.co):
    mkdir(args.co)
if not exists(args.no):
    mkdir(args.no)
if not exists(args.pn):
    mkdir(args.pn)

dirname = "dataset"
files = [f for f in listdir(dirname)]

def preprocess(img):
    return img.resize((args.image_width, args.image_height))

for f in files:
   img = Image.open(join(dirname, f))
   img = preprocess(img)
   if "co" in f.lower():
       img.save(join(args.co, f))
   elif "no" in f.lower():
       img.save(join(args.no, f))
   elif "pn" in f.lower():
       img.save(join(args.pn, f))
   else:
       print("ERROR: Unexpected file encountered, Filename: " + f)
       exit()
