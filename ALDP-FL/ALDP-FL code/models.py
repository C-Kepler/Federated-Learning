
import torch 
from torchvision import models
import math

def get_model(name="vgg16", pretrained=False):
	if name == "resnet18":
		model = models.resnet18(pretrained=pretrained)
	elif name == "resnet50":
		model = models.resnet50(pretrained=pretrained)	
	elif name == "densenet121":
		model = models.densenet121(pretrained=pretrained)		
	elif name == "alexnet":
		model = models.alexnet(pretrained=pretrained)
	elif name == "vgg16":
		model = models.vgg16(pretrained=pretrained)
	elif name == "vgg19":
		model = models.vgg19(pretrained=pretrained)
	elif name == "inception_v3":
		model = models.inception_v3(pretrained=pretrained)
	elif name == "googlenet":		
		model = models.googlenet(pretrained=pretrained)
	if torch.cuda.is_available():
		return model.cuda()
	else:
		return model 
		
def model_norm(model_1, model_2):
	squared_sum = 0
	for name, layer in model_1.named_parameters():
	#	print(torch.mean(layer.data), torch.mean(model_2.state_dict()[name].data))
		squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
	return math.sqrt(squared_sum)


def layer_norm2(model_1, model_2):
	layerdiff = dict()
	for name, params in model_1.items():
		params_float = params.float()
		params_model_2_float = model_2[name].float()
		#print("data1:",params_float,"\ndata2:",params_model_2_float)

		diff = params_model_2_float - params_float
		#print("diff:",diff)
		layerdiff[name] = torch.norm(diff, p=2).item()

	return layerdiff