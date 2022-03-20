import torch.nn.functional
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import random

class german_traffic(Dataset):

	def __init__(self, path_dataset, transforms=None, data="train"):
		self.path = path_dataset
		if data=="train":
			self.data = self._dataloader(self.path, "Train")
		else:
			self.data = self._dataloader(self.path, "Test")
		self.transforms = transforms

	def __getitem__(self, item):
		id, img_path = self.data[item]
		img = read_image(self.path +"/"+ img_path)
		if self.transforms:
			img = self.transforms(img)
		# id = torch.nn.functional.one_hot(torch.tensor(int(id)), 44)
		return img.float(), int(id)

	def __len__(self):
		return len(self.data)

	def _dataloader(self, dataPath, sets="Train"):
		'''Get the dataset and load the paths and index'''
		first = True
		images = []
		with open(dataPath +"/"+ sets+ ".csv") as f:
			for line in f.readlines():
				if first:
					first = False
					continue
				_, _, _, _, _, _, classId, path = line.strip().split(",")
				images.append((classId, path))
		return images

	def imshow(self):
		fig, ax = plt.subplots(5, 5)
		for i in range(5):
			for j in range(5):
				id, img_path = random.choice(self.data)
				img = read_image(self.path +"/"+ img_path)
				if self.transforms:
					img = self.transforms(img)
				print(id)
				ax[i][j].imshow(img.permute(1, 2, 0))
		plt.show()

if __name__=="__main__":
	transforms = torchvision.transforms.Resize((32,32))
	germanData = german_traffic("/home/bhargav/german_traffic/archive",transforms=transforms, data="train")
	# germanData.imshow()
	print(len(germanData))
	# for i in range(10):
	# 	img, id = germanData[i]
	# 	print(id)