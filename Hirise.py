from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

#root for the path
#classes (num -> labels)

#samples (im_name -> labels)
#getitem (image -> labels)

#transform (image ->)

class HiRise(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = self._load_class_map() # {id:name}

        # proccessing the (image_file_name, label, class_num) form the txt file
        self.samples = [] #list of (path,int)
        with open(self.root / "labels-map-proj_v3_2_train_val_test.txt") as f:
            for line in f:
                line = line.strip() # drops the '\n' and any spaces at ends

                if not line:    # skip the blanks
                    continue 

                filename, label, split_name = line.split() # this will break the line into three pieces

                if split_name != split:
                    continue

                self.samples.append((self.root/'map-proj-v3_2'/filename,int(label)))


    def _load_class_map(self):
        classes = {}
        with open(self.root/'landmarks_map-proj-v3_2_classmap.csv') as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                class_num, class_name = line.split(",")
                classes[int(class_num)] = class_name
            return classes

    def __len__(self):
        return len(self.samples)

    # this class is made for the loader to get the items
    def __getitem__(self,i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label