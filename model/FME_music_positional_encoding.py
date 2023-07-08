import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


class Fundamental_Music_Embedding(nn.Module):
	def __init__(self, d_model, base, device='cpu'):
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

	def transform_by_delta_pos_v2(self, inp, delta_pos):
		#fast version, no need to use block diagonal matrix
		#transpose one token to another in the embedding space
		
		batch, length = int(inp.shape[0]), int(inp.shape[1])
		raw = self.FMS(delta_pos)
		wk_phi_1 = torch.reshape(raw,[batch*length*int(self.d_model/2), 2]) #[d_mod/2, 2] -->batch* len* d_mod/2, 2
		wk_phi_1_rev=wk_phi_1*torch.tensor([-1., 1.]).to(self.device)[None, ...] # (batch*len*d_mod/2, 2) * (1, 2)
		wk_phi_2 = torch.flip(wk_phi_1, dims = [-1]) ##[d_mod/2, 2] --># (batch*len*d_mod/2, 2)
	
		wk_phi1_2 = torch.cat((wk_phi_2, wk_phi_1_rev), axis = -1) #[dmod/2, 4] # (batch* len* d_mod/2, 4)
		wk_phi1_2_rehsaped = torch.reshape(wk_phi1_2, [batch*length*int(self.d_model/2), 2, 2]) #[dmod/2, 2, 2] --># (batch* len*d_mod/2, 2, 2) we want -->1*3*4*4
		transformation_matrix = wk_phi1_2_rehsaped 
		
		inp -= self.translation_bias[:, None, :]

		reshaped = torch.reshape(inp, (batch*length*int(self.d_model/2), 2,1))
		out = torch.matmul(transformation_matrix, 
							reshaped) #(batch* len*d_mod/2, 2, 2) * (batch*len*d_mod, 1, 2)

		out = torch.reshape(out, (batch, length, self.d_model))
		out += self.translation_bias[:, None, :]
		return out

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
	
	def FMS(self, delta_pos):
		if delta_pos.dim()==1:
			delta_pos = delta_pos[None, ..., None] # len ==> batch, len
		if delta_pos.dim()==2:
			delta_pos = delta_pos[ ..., None] # batch, len ==> batch, len, 1
		if delta_pos.dim()==3:
			b_size = delta_pos.shape[0]
			len_q = delta_pos.shape[1]
			len_k = delta_pos.shape[2]
			delta_pos = delta_pos.reshape((b_size, len_q*len_k, 1))# batch, len, len ==> batch, len*len, 1
		
		raw = delta_pos*self.angles
		raw[:, :, 0::2] = torch.sin(raw.clone()[:, :, 0::2])
		raw[:,:,1::2] = torch.cos(raw.clone()[:,:,1::2])

		if delta_pos.dim()==3:
			raw = raw.reshape((b_size, len_q, len_k, -1))# batch, len, len ==> batch, len*len, 1
		return raw.to(torch.float32).to(self.device)

	def decode(self, embedded):
		embedded -= self.translation_bias[:, None, :]

		decoded_dim = (torch.asin(embedded)/self.angles[:, None, :]).to(torch.float32)
		if self.d_model/2 %2 == 0:
			decoded = decoded_dim[:, :, int(self.d_model/2)]

		elif self.d_model/2 %2 == 1:	
			decoded = decoded_dim[:, :, int(self.d_model/2+1)]

		return decoded 

	def decode_tps(self, embedded):
		decoded_dim = (torch.asin(embedded)/self.angles[:, None,None, :]).to(torch.float32)
		if self.d_model/2 %2 == 0:
			decoded = decoded_dim[:, :, :, int(self.d_model/2)]

		elif self.d_model/2 %2 == 1:	
			decoded = decoded_dim[:, :, :, int(self.d_model/2+1)]

		return decoded 

class Music_PositionalEncoding(nn.Module):

	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device = 'cuda:0'):
		super().__init__()

		self.dropout = nn.Dropout(p=dropout)
		self.global_time_embedding = Fundamental_Music_Embedding(d_model = d_model, base=10001, device = device).cuda()
		self.modulo_time_embedding = Fundamental_Music_Embedding(d_model = d_model, base=10001, device = device).cuda()

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
		
		modulo_timing = dur_onset_cumsum % 16
		modulo_timing_embedding = self.modulo_time_embedding(modulo_timing)
		inp += modulo_timing_embedding
		return self.dropout(inp)
