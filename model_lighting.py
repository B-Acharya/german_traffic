import pytorch_lightning as pl
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import os
import torch.optim as optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from argparse import ArgumentParser
from dataset import german_traffic
from model import model_conv
import torch.utils.data as data
import torchvision.transforms

class lighitngModule(pl.LightningModule):
	def __init__(self, model, optimizer_name, lr=None, weigh_decay=None):
		"""
		Inputs:
			model_name - Name of the model/CNN to run. Used for creating the model (see function below)
			model_hparams - Hyperparameters for the model, as dictionary.
			optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
			optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
		"""
		super().__init__()

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

		self.loss_module = nn.CrossEntropyLoss()

	def forward(self, x):
		return self.layers(x)

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, batch, batch_idx):
		# "batch" is the output of the training data loader.
		imgs, labels = batch
		preds = self.model(imgs)
		loss = self.loss_module(preds, labels)
		acc = (preds.argmax(dim=-1) == labels).float().mean()
		# Logs the accuracy per epoch to tensorboard (weighted average over batches)
		self.log('train_acc', acc, on_step=False, on_epoch=True)
		self.log('train_loss', loss)
		return loss  # Return tensor to call ".backward" on

	def validation_step(self, batch, batch_idx):
		imgs, labels = batch
		preds = self.model(imgs).argmax(dim=-1)
		acc = (labels == preds).float().mean()
		# By default logs it per epoch (weighted average over batches)
		self.log('val_acc', acc)

	def test_step(self, batch, batch_idx):
		imgs, labels = batch
		preds = self.model(imgs).argmax(dim=-1)
		acc = (labels == preds).float().mean()
		# By default logs it per epoch (weighted average over batches), and returns it afterwards
		self.log('test_acc', acc)


def train_model(train_loader, test_loader,val_loader, lr=None, weight_decay=None, save_name=None, device=None, CHECKPOINT_PATH=None):
	"""
	Inputs:
		model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
		save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
	"""
	save_name = "model_1"

	# Create a PyTorch Lightning trainer with the generation callback
	trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),                          # Where to save models
						 gpus=1 if str(device)=="cuda:0" else 0,                                             # We run on a single GPU (if possible)
						 max_epochs=1,                                                                     # How many epochs to train for if no patience is set
						 callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
									LearningRateMonitor("epoch")],                                           # Log learning rate every epoch
						 progress_bar_refresh_rate=1)                                                        # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
	trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
	trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

	# Check whether pretrained model exists. If yes, load it and skip training
	pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
	if os.path.isfile(pretrained_filename):
		print(f"Found pretrained model at {pretrained_filename}, loading...")
		model = lighitngModule.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
	else:
		pl.seed_everything(42) # To be reproducable
		model = lighitngModule()
		trainer.fit(model, train_loader, val_loader)
		model = lighitngModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

	# Test best model on validation and test set
	val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
	test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
	result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

	return model, result

def main(args, device):
	transforms = torchvision.transforms.Resize((32, 32))
	germanDataTrain = german_traffic("/home/bhargav/german_traffic/archive", transforms=transforms, data="train")
	germanDataTest = german_traffic("/home/bhargav/german_traffic/archive", transforms=transforms, data="Test")
	validation_split = .2
	shuffle_dataset = True
	random_seed = 42
	# Creating data indices for training and validation splits:
	dataset_size = len(germanDataTrain)
	indices = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))
	if shuffle_dataset:
		np.random.seed(random_seed)
		np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]

	# Creating PT data samplers and loaders:
	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)
	train_loader = torch.utils.data.DataLoader(germanDataTrain, batch_size=32,
											   sampler=train_sampler)
	validation_loader = torch.utils.data.DataLoader(germanDataTrain, batch_size=32,
													sampler=valid_sampler)
	test_data = data.DataLoader(germanDataTest, batch_size=32)
	model = model_conv()
	model_1, model_1_results = train_model(model_name=model,train_loader=train_loader,val_loader=validation_loader,test_loader=test_data,device=device,lr=args.lr,weight_decay=1e-3, CHECKPOINT_PATH="~/german_traffic/checkpoints")

if __name__=="__main__":
	parser = ArgumentParser()
	parser.add_argument("--epochs", default=15)
	parser.add_argument("--lr", default=1e-3)
	parser.add_argument("--save_name", default=None)
	parser.add_argument("--CHECKPOINT_PATH", default="/home/bhargav/rPPG_Deeplearning/models/")
	parser.add_argument("--dataset",default="cohface")
	parser.add_argument("--datase_path", default="/home/bhargav/rPPG_Deeplearning/src/data/")
	args = parser.parse_args()
	device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
	main(args, device)
