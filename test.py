import json 
import cv2
import os
import numpy as np

results_dir = "./save_folder/"
file_names = os.listdir(results_dir)
file_names = [f for f in file_names if f.endswith(".json")]
for file_name in file_names:
    img_name = file_name.split(".")[0]
    res = json.load(open(results_dir + file_name))
    boxes = res["boxes"]
    img = cv2.imread(results_dir + img_name + ".jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        for box in boxes:
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255,0,0), 2)
    except:
        pass
    cv2.imshow("as", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
