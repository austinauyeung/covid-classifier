# Metadata file for NIH Chest X-rays
md_file = "Data_Entry_2017.csv"
trainfile = "dataset.txt"

mdf = open(md_file, 'r')

card = []
norm = []

lines = mdf.readlines()
for line in lines:
    if "Cardiomegaly" in line:
        card.append(line.split(',')[0])
    elif "No Finding" in line:
        norm.append(line.split(',')[0])
mdf.close()

# Take ~20 % and put it in test, rest in train
N = len(card) 
M = int(0.8*N)

card_train = card[:M-1]
card_test = card[M:N]

norm_train = norm[:M-1]
norm_test = norm[M:N]

def write_dataset(datafile, data_card, data_norm):
    datafile = open(datafile, 'w')
    datafile.write("#Cardiomegaly\n")
    for file in data_card:
        datafile.write("./data/images/" + file + '\n')
    datafile.write("#Normal\n")
    for file in data_norm:
        datafile.write("./data/images/" + file + '\n')
    datafile.close()

write_dataset("train.txt", card_train, norm_train)
write_dataset("test.txt", card_test, norm_test)