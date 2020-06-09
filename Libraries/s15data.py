import os
import random
def dataset():
  path = "/content/drive/My Drive/EVA4/data/bgimages/"
  bg = []
  for p in os.listdir(path):
    for i in range(1000):
      bg.append(path+p)
  bg.sort()

  fgbgpath = "/content/output/images/"
  fgbg= []
  for p in os.listdir(fgbgpath):
    fgbg.append(fgbgpath+p)
  fgbg.sort()

  maskpath = "/content/output/masks/"
  mask = []
  for p in os.listdir(maskpath):
    mask.append(maskpath+p)
  mask.sort()

  depthpath = "/content/output/depth/"
  depth= []
  for p in os.listdir(depthpath):
    depth.append(depthpath+p)
  depth.sort()

  data = list(zip(bg[:200], fgbg[:200], mask[:200], depth[:200]))
  random.shuffle(data)
  return data