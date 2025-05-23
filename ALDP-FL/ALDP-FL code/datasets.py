

import torch 
from torchvision import datasets, transforms

def get_dataset(dir, name):

	if name=='MNIST':
		transform = transforms.Compose([
			transforms.Resize((224, 224)),  # Adjust the image size to 224x224
			transforms.Grayscale(num_output_channels=3),  # Convert grayscale image to 3 channels
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,))
		])
		train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transform)
		eval_dataset = datasets.MNIST(dir, train=False, transform=transform)
	elif name == 'FashionMNIST':
		transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.Grayscale(num_output_channels=3),
			transforms.ToTensor(),
		])
		train_dataset = datasets.FashionMNIST(dir, train=True, download=True, transform=transform)
		eval_dataset = datasets.FashionMNIST(dir, train=False, transform=transform)


	elif name=='cifar':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		
		train_dataset = datasets.CIFAR10(dir, train=True, download=True,
										transform=transform_train)
		eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
		
	
	return train_dataset, eval_dataset