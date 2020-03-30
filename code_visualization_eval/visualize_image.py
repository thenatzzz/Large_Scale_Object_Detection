'''Visualize images in panels'''

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import figure
import numpy as np

path = "positive_test"+"\\"
filenames=[path+'P03320.jpg',path+'P03320.jpg',path+'P03320.jpg',path+'P03320.jpg',\
path+'P03320.jpg',path+'P03320.jpg',path+'P03320.jpg',path+'P03320.jpg',path+'P03320.jpg']

# settings
h, w = 10, 10        # for raster image
nrows, ncols = 3, 3  # array of sub-plots
figsize = [6, 8]     # figure size, inches

# create figure (fig), and array of axes (ax)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,gridspec_kw={'hspace': 0, 'wspace': 0}, constrained_layout=True)

# plot simple raster image on each sub-plot
for i, axi in enumerate(ax.flat):
    # i runs from 0 to (nrows*ncols-1)
    # axi is equivalent with ax[rowid][colid]
    with open(filenames[i],'rb') as f:
        img=Image.open(f)
#         ax[i%3][i//3].imshow(image)
        # img = np.random.randint(10, size=(h,w))
        axi.imshow(img, alpha=0.90)
    # get indices of row/column
    rowid = i // ncols
    colid = i % ncols
    # write row/col indices as axes' title for identification
    axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))

for ax_ in ax.flat:
    ax_.label_outer()

plt.tight_layout(True)
plt.show()
