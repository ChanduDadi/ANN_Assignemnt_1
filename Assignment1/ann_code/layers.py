"""THWS/MAI/ANN - Assignment 1 - feed forward network layers

Created: Magda Gregorova, 9/5/2024
"""


import torch


class Linear:
	"""Apply linear transformation to input data y = X W^T + b.

	Attributes:
		W: torch.tensor of shape (out_feartures, in_features) - weight matrix
		b: torch.tensor of shape (1, out_features) - bias vector
		ins: torch.tensor of shape (num_instances, in_features) - input data
		outs: torch.tensor of shape (n, out_features) - output data
		W.g: torch.tensor of shape (out_feartures, in_features) - weight matrix global gradients
		b.g: torch.tensor of shape (1, out_features) - bias vector global gradients
		ins.g: torch.tensor of shape (num_instances, in_features) - input data global gradients
	"""

	def __init__(self, W, b):
		"""Initiate instances with weight and bias attributes.

		Arguments:
			W: torch.tensor of shape (out_feartures, in_features) - weight matrix
			b: torch.tensor of shape (1, out_features) - bias vector
		""" 
		################################################################################
		### START OF YOUR CODE                                                         #
		### TODO: implement the init step.                                             #	
		################################################################################
		self.W = W
		self.b = b
		################################################################################
		### END OF YOUR CODE                                                           #
		################################################################################

	def forward(self, ins):
		"""Forward pass through linear transformation. Populates ins and outs attributes.

		Arguments:
			ins: torch.tensor of shape (num_instances, in_features) - input data

		Returns:
			torch.tensor of shape (num_instances, out_features) - output data
		""" 
		################################################################################
		### START OF YOUR CODE                                                         #
		### TODO: implement the forward pass.										   #	
		################################################################################
		self.ins = ins
		self.outs = torch.matmul(ins, self.W.t()) + self.b
		################################################################################
		### END OF YOUR CODE                                                           #
		################################################################################
		return self.outs


	def backward(self, gout):
		"""Backward pass through linear transformation. Populates W.g, b.g and ins.g attributes.

		Arguments:
			gout: torch.tensor of shape (num_instances, out_features) - upstream gradient

		Returns:
			torch.tensor of shape (num_instances, num_dims) - input data global gradients
		""" 
		################################################################################
		### START OF YOUR CODE                                                         #
		### TODO: implement the backward pass.										   #	
		################################################################################
		self.W.g = torch.matmul(gout.t(), self.ins)
		self.b.g = torch.sum(gout, dim=0, keepdim=True)
		self.ins.g = torch.matmul(gout, self.W)
		################################################################################
		### END OF YOUR CODE                                                           #
		################################################################################
		return self.ins.g


class Relu:
	"""Apply relu non-linearity x = max(0, x).

	Attributes:
		ins: torch.tensor of shape (num_instances, num_dims) - input data
		outs: torch.tensor of shape (num_instances, num_dims) - output data
		ins.g: torch.tensor of shape (num_instances, num_dims) - input data global gradients
	"""

	def forward(self, ins):
		"""Forward pass through relu. Populates ins and outs attributes.

		Arguments:
			ins: torch.tensor of shape (num_instances, num_dims) - input data

		Returns:
			torch.tensor of shape (num_instances, num_dims) - output data
		""" 
		################################################################################
		### START OF YOUR CODE                                                         #
		### TODO: implement the forward pass.										   #	
		################################################################################
		self.ins = ins
		self.outs = torch.max(ins, torch.tensor(0.))
		################################################################################
		### END OF YOUR CODE                                                           #
		################################################################################
		return self.outs

	def backward(self, gout):
		"""Backward pass through relu. Populates ins.g attributes.

		Arguments:
			gout: torch.tensor of shape (num_instances, num_dims) - upstream gradient

		Returns:
			torch.tensor of shape (num_instances, num_dims) - input data global gradients
		""" 
		################################################################################
		### START OF YOUR CODE                                                         #
		### TODO: implement the backward pass.										   #	
		################################################################################
		self.ins.g = gout * (self.ins > 0).float()
		################################################################################
		### END OF YOUR CODE                                                           #
		################################################################################
		return self.ins.g


class Model():
	"""Neural network model.

	Attributes:
		layers: list of NN layers in the order of the forward pass from inputs to outputs
	"""

	def __init__(self, layers):
		"""Initiate model instance all layers. 

		Layers are expected to be instances of Linear and Relu classes.
		The shall be passed to Model instances as a list in the correct order of forward execution.

		Arguments:
			layers: list of layer instances		
		"""
		self.layers = layers

	def forward(self, ins):
		"""Forward pass through model. 

		Arguments:
			ins: torch.tensor of shape (num_instances, in_features) - input data

		Returns:
			torch.tensor of shape (n, out_features) - model predictions
		""" 
		outs = ins
		for layer in self.layers:
			outs = layer.forward(outs)
		return outs

	def backward(self, gout):
		"""Backward pass through model

		Arguments:
			gout: torch.tensor of shape (num_instances, out_features) - gradient of loss with respect to predictions

		Returns:
			torch.tensor of shape (n, in_features) - gradient with respect to forward inputs
		""" 
		for layer in reversed(self.layers):
			gout = layer.backward(gout)
		return gout

