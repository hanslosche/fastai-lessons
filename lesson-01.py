from fastai.imports import *
from fastai.vision import *
import torch

PATH = "/Users/hlos/testing_unit/"
sz = 224
torch.cuda.is_available()

fnames = np.array([f'train/{f}' for f in sorted(os.listdir(f'{PATH}train'))])
labels = np.array([( 0 if 'cat' in fname else 1 ) for fname in fnames])
print(labels)
print(fnames)


img = plt.imread(f'{PATH}{fnames[0]}')
plt.imshow(img)
plt.show()
