import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

class Fundamental_Music_Embedding(nn.Module):
	def __init__(self, d_model, base, device='cuda:0'):
		super().__init__()
		self.d_model = d_model
		self.device = device
		self.base = base
		
		translation_bias = torch.rand((1, self.d_model), dtype = torch.float32)
		translation_bias = nn.Parameter(translation_bias, requires_grad=True)
		self.register_parameter("translation_bias", translation_bias)

		i = torch.arange(d_model)
		angle_rates = 1 / torch.pow(self.base, (2 * (i//2)) / d_model)
		angle_rates = angle_rates[None, ... ].to(self.device)
		angles = nn.Parameter(angle_rates, requires_grad=True)
		self.register_parameter("angles", angles)

	def __call__(self, inp):
		inp = inp[..., None] #pos (batch, num_pitch, 1)
		angle_rads = inp*self.angles #(batch, num_pitch)*(1,dim)

		# apply sin to even indices in the array; 2i
		angle_rads[:, :, 0::2] = torch.sin(angle_rads.clone()[:, : , 0::2])

		# apply cos to odd indices in the array; 2i+1
		angle_rads[:, :, 1::2] = torch.cos(angle_rads.clone()[:, :, 1::2])

		pos_encoding = angle_rads.to(torch.float32)

		if self.translation_bias.size()[-1]!= self.d_model:
			translation_bias = self.translation_bias.repeat(1, 1,int(self.d_model/2))
		else:
			translation_bias = self.translation_bias
		pos_encoding += translation_bias
		
		return pos_encoding
	

class Music_PositionalEncoding(nn.Module):

	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device = 'cuda:0'):
		super().__init__()

		self.dropout = nn.Dropout(p=dropout)
		self.global_time_embedding = Fundamental_Music_Embedding(d_model = d_model, base=10001, device = device)
		
		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)
	
	def forward(self, inp,　dur_onset_cumsum = None):

		pe_index = self.pe[:inp.size(1)] #[seq_len, batch_size, embedding_dim]
		pe_index = torch.swapaxes(pe_index, 0, 1) #[batch_size, seq_len, embedding_dim]
		inp += pe_index
		
		global_timing = dur_onset_cumsum
		global_timing_embedding = self.global_time_embedding(global_timing)
		inp += global_timing_embedding
		
		return self.dropout(inp)
