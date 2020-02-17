import os
import shutil

files = os.listdir("07Dataset")
names = []
print(len(files))
for file in files:
    names.append(file.split(".")[0])
names = set(names)
i =0 
for name in names:
    if int(name) < 100 and int (name) > 81:
        shutil.copyfile("07Dataset/" + name + ".jpg", "val/" + name+ ".jpg")
        shutil.copyfile("07Dataset/" + name + ".json", "val/" + name+ ".json")
    else:
        shutil.copyfile("07Dataset/" + name + ".jpg", "train/" + name + ".jpg")
        shutil.copyfile("07Dataset/" + name + ".json", "train/" + name+ ".json")
    i+=1