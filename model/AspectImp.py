import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


class AspectImp(nn.Module):

	def __init__(self, logger, args):

		super(AspectImp, self).__init__()

		self.logger = logger
		self.args = args


		# Matrix for Interaction between User Aspect-level Representations & Item Aspect-level Representations 
		# This is a learnable (h1 x h1) matrix, i.e. User Aspects - Rows, Item Aspects - Columns
		# W_s
		self.W_a = nn.Parameter(torch.Tensor(self.args.h1, self.args.h1), requires_grad = True)

		# User "Projection": A (h1 x h2) weight matrix, and a (h2 x 1) vector
		# W_x
		self.W_u = nn.Parameter(torch.Tensor(self.args.h2, self.args.h1), requires_grad = True)
		# v_x
		self.w_hu = nn.Parameter(torch.Tensor(self.args.h2, 1), requires_grad = True)

		# Item "Projection": A (h1 x h2) weight matrix, and a (h2 x 1) vector
		# W_y
		self.W_i = nn.Parameter(torch.Tensor(self.args.h2, self.args.h1), requires_grad = True)
		# v_y
		self.w_hi = nn.Parameter(torch.Tensor(self.args.h2, 1), requires_grad = True)


		# Initialize all weights using random uniform distribution from [-0.01, 0.01]
		self.W_a.data.uniform_(-0.01, 0.01)

		self.W_u.data.uniform_(-0.01, 0.01)
		self.w_hu.data.uniform_(-0.01, 0.01)

		self.W_i.data.uniform_(-0.01, 0.01)
		self.w_hi.data.uniform_(-0.01, 0.01)


	def forward(self, userAspRep, itemAspRep, verbose = 0):

		if(verbose > 0):

			tqdm.write("[Input to AIE] userAspRep: {}".format( userAspRep.size() ))
			tqdm.write("[Input to AIE] itemAspRep: {}".format( itemAspRep.size() ))

		userAspRepTrans = torch.transpose(userAspRep, 1, 2)
		itemAspRepTrans = torch.transpose(itemAspRep, 1, 2)
		if(verbose > 0):
			tqdm.write("\nuserAspRepTrans: {}".format( userAspRepTrans.size() ))
			tqdm.write("itemAspRepTrans: {}".format( itemAspRepTrans.size() ))

		diff = torch.abs(userAspRep[:, :, None, :] - itemAspRep[:, None, :, :])

		if (verbose > 0):
			tqdm.write("diff: {}".format(diff.size()))

		# Calculate similarity: 1 / (1 + |p_k^u - q_j^i|)
		similarityMatrix = 1 / (1 + diff.sum(dim=-1))  # Summing over the hidden_size dimension

		if verbose > 0:
			tqdm.write("similarityMatrix: {}".format(similarityMatrix.size()))
		# P_u * W_x
		H_u_1 = torch.matmul(self.W_u, userAspRepTrans)
		# Q_i * W_y
		H_u_2 = torch.matmul(self.W_i, itemAspRepTrans)
		# S^T * (Q_i * W_y)
		H_u_2 = torch.matmul(H_u_2, torch.transpose(similarityMatrix, 1, 2))

		H_u = H_u_1 + H_u_2

		# Non-Linearity: ReLU
		H_u = F.relu(H_u)

		# User Aspect-level Importance
		# \beta_u = softmax(H_u * v_x)
		userAspImpt = torch.matmul(torch.transpose(self.w_hu, 0, 1), H_u)
		if(verbose > 0):
			tqdm.write("\nuserAspImpt: {}".format( userAspImpt.size() ))

		# User Aspect-level Importance: (bsz x 1 x num_aspects) -> (bsz x num_aspects x 1)
		userAspImpt = torch.transpose(userAspImpt, 1, 2)
		if(verbose > 0):
			tqdm.write("userAspImpt: {}".format( userAspImpt.size() ))

		userAspImpt = F.softmax(userAspImpt, dim = 1)
		if(verbose > 0):
			tqdm.write("userAspImpt: {}".format( userAspImpt.size() ))

		# User Aspect-level Importance: (bsz x num_aspects x 1) -> (bsz x num_aspects)
		userAspImpt = torch.squeeze(userAspImpt, 2)
		if(verbose > 0):
			tqdm.write("userAspImpt: {}".format( userAspImpt.size() ))

		H_i_1 = torch.matmul(self.W_i, itemAspRepTrans)
		H_i_2 = torch.matmul(self.W_u, userAspRepTrans)
		H_i_2 = torch.matmul(H_i_2, similarityMatrix)
		H_i = H_i_1 + H_i_2

		# Non-Linearity: ReLU
		H_i = F.relu(H_i)

		# Item Aspect-level Importance
		itemAspImpt = torch.matmul(torch.transpose(self.w_hi, 0, 1), H_i)
		if(verbose > 0):
			tqdm.write("\nitemAspImpt: {}".format( itemAspImpt.size() ))

		# Item Aspect-level Importance: (bsz x 1 x num_aspects) -> (bsz x num_aspects x 1)
		itemAspImpt = torch.transpose(itemAspImpt, 1, 2)
		if(verbose > 0):
			tqdm.write("itemAspImpt: {}".format( itemAspImpt.size() ))

		itemAspImpt = F.softmax(itemAspImpt, dim = 1)
		if(verbose > 0):
			tqdm.write("itemAspImpt: {}".format( itemAspImpt.size() ))

		# Item Aspect-level Importance: (bsz x num_aspects x 1) -> (bsz x num_aspects)
		itemAspImpt = torch.squeeze(itemAspImpt, 2)
		if(verbose > 0):
			tqdm.write("itemAspImpt: {}".format( itemAspImpt.size() ))

		if(verbose > 0):
			tqdm.write("\n[Output of AIE] userAspImpt (i.e. the User Aspect-level Importance): {}".format( userAspImpt.size() ))
			tqdm.write("[Output of AIE] itemAspImpt (i.e. the Item Aspect-level Importance): {}".format( itemAspImpt.size() ))
			tqdm.write("============================== ================================== ==============================\n")

		return userAspImpt, itemAspImpt

