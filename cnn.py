from Hirise import HiRise

ds = HiRise("data/hirise-map-proj-v3_2/hirise-map-proj-v3_2", split="train")
print(len(ds))
print(ds.samples[0])
print(ds.samples[0][0].exists())
print(ds.classes)
img, label = ds[0]
print(type(img), img.size, img.mode, label, ds.classes[label])