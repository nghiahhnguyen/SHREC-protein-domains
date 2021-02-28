from torch import nn


def remove_final_layer(model):
	name, module = list(model.named_modules())[-1]
	setattr(model, name, nn.Identity())