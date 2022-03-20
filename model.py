import torch
import torch.nn as nn
import torchvision.transforms
from dataset import german_traffic

class model_conv(nn.Module):

	def __init__(self):
		super(model_conv, self).__init__()
		self.layers = torch.nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=(5,5)),
			nn.ReLU(),
			nn.Conv2d(32,64,kernel_size=(3,3)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Dropout(p=0.25),
			nn.Conv2d(64,64,kernel_size=(3,3)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2)),
			nn.Dropout(p=0.25),
			nn.Flatten(),
			nn.Linear(64*5*5,256),
			nn.ReLU(),
			nn.Dropout(p=0.25),
			nn.Linear(256,43),
			nn.Softmax()
		)

	def forward(self, x):
		return self.layers(x)

if __name__=="__main__":
	model= model_conv()
	transforms = torchvision.transforms.Resize((30,30))
	germanData = german_traffic("/home/bhargav/german_traffic/archive",transforms=transforms, data="train")
	img, id = germanData[0]
	output = model(img.float().unsqueeze(0))
