from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


root_dir = 'work_test'

img_width, img_height = 64, 64

model = load_model('simpsons_model_trained')

chardict = {
 'bart': 0,
 'homer': 1,
 'lisa': 2,
 'marge': 3
 }

rows = 2
cols = 3
fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(7, 7))
fig.suptitle('work_test images', fontsize=20)
count=0
for i in range(rows):
    for j in range(cols):
        all_files = os.listdir(root_dir)
        imgpath = os.path.join(root_dir, all_files[count])
        img = Image.open(imgpath)
        ax[i][j].imshow(img)
        img = img.convert("RGB")
        img = img.resize((img_width, img_height), Image.ANTIALIAS)
        img = img_to_array(img)
        img = img/255.0
        img = img.reshape((1,) + img.shape)
        pred = model.predict(img, batch_size = 32)
        pred = pd.DataFrame(np.transpose(np.round(pred, decimals = 3)))
        pred = pred.nlargest(n = 3, columns = 0)
        pred['char'] = [list(chardict.keys())[list(chardict.values()).index(x)] for x in pred.index]
        charstr = ''
        for k in range(0,3):
            if k < 2:
                charstr = charstr+str(pred.iloc[k,1])+': '+str(pred.iloc[k,0])+'\n'
            else:
                charstr = charstr+str(pred.iloc[k,1])+': '+str(pred.iloc[k,0])
        ec = (0, .8, .1)
        fc = (0, .9, .2)
        count = count + 1
        ax[i][j].text(0, -10, charstr, size=10, rotation=0,
                ha="left", va="top", 
                bbox=dict(boxstyle="round", ec=ec, fc=fc, alpha = 0.7))
plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
