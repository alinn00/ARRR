import torch
import torch.nn as nn
import torch.nn.functional as F

from .utilities import to_var

from tqdm import tqdm


class AspectRep(nn.Module):

	def __init__(self, logger, args, num_users, num_items):

		super(AspectRep, self).__init__()

		self.logger = logger
		self.args = args

		self.num_users = num_users
		self.num_items = num_items

		self.aspProj = nn.Parameter(torch.Tensor(self.args.num_aspects, self.args.word_embed_dim, self.args.h1), requires_grad = True)
		self.aspProj.data.uniform_(-0.01, 0.01)

		self.l1 = nn.Linear(self.args.d1, self.args.h1)

		# Create separate embedding layers for users and items
		self.user_embedding = nn.Embedding(num_users, self.args.d1)
		self.item_embedding = nn.Embedding(num_items, self.args.d1)

		self.user_embedding.weight.requires_grad = True
		self.item_embedding.weight.requires_grad = True

		# Initialize the weights of the embedding layer
		self.user_embedding.weight.data.uniform_(-0.01, 0.01)
		self.item_embedding.weight.data.uniform_(-0.01, 0.01)

	def forward(self, args, batch_docIn, batch_id, nums, verbose = 0):

		if(verbose > 0):
			tqdm.write("[Input] batch_docIn: {}".format( batch_docIn.size() ))

		# Loop over all aspects
		lst_batch_aspAttn = []
		lst_batch_aspRep = []
		for a in range(self.args.num_aspects):

			if(verbose > 0 and a == 0):
				tqdm.write("\nAs an example, for <Aspect {}>:\n".format( a ))

			if(nums == self.num_items):
				id_vector = self.item_embedding(batch_id).cuda()
			else:
				id_vector = self.user_embedding(batch_id).cuda()
			qu = F.relu(self.l1(id_vector)).cuda()
			qu = qu.unsqueeze(-1)

			batch_aspProjDoc = torch.matmul(batch_docIn, self.aspProj[a])

			if(verbose > 0 and a == 0):
				tqdm.write("\tbatch_docIn: {}".format( batch_docIn.size() ))
				tqdm.write("\tself.aspProj[{}]: {}".format( a, self.aspProj[a].size() ))
				tqdm.write("\tbatch_aspProjDoc: {}".format( batch_aspProjDoc.size() ))

			if(self.args.ctx_win_size == 1):

				attention_scores = torch.matmul(batch_aspProjDoc, qu).squeeze(-1)
				# Apply softmax to get attention weights
				attention_weights = F.softmax(attention_scores, dim=1)
				if(verbose > 0 and a == 0):
					tqdm.write("\n\tbatch_aspAttn: {}".format( attention_weights.size() ))
			else:

				# Pad the document
				qu = qu.repeat(1, self.args.ctx_win_size, 1)
				pad_size = int((self.args.ctx_win_size - 1) / 2)
				batch_aspProjDoc_padded = F.pad(batch_aspProjDoc, (0, 0, pad_size, pad_size), "constant", 0)
				if (verbose > 0 and a == 0):
					tqdm.write("\n\tbatch_aspProjDoc_padded [PADDED; Pad Size: {}]: {}".format(pad_size,
																							   batch_aspProjDoc_padded.size()))
				batch_aspProjDoc_padded = batch_aspProjDoc_padded.unfold(1, self.args.ctx_win_size, 1)
				if (verbose > 0 and a == 0):
					tqdm.write("\tbatch_aspProjDoc_padded: {}".format(batch_aspProjDoc_padded.size()))
				batch_aspProjDoc_padded = torch.transpose(batch_aspProjDoc_padded, 2, 3)
				if (verbose > 0 and a == 0):
					tqdm.write("\tbatch_aspProjDoc_padded: {}".format(batch_aspProjDoc_padded.size()))
				batch_aspProjDoc_padded = batch_aspProjDoc_padded.contiguous().view(-1, self.args.max_doc_len,
																					self.args.ctx_win_size * self.args.h1)
				if (verbose > 0 and a == 0):
					tqdm.write("\tbatch_aspProjDoc_padded: {}".format(batch_aspProjDoc_padded.size()))

				attention_scores = torch.matmul(batch_aspProjDoc_padded, qu).squeeze(-1)
				attention_weights = F.softmax(attention_scores, dim=1)
				if (verbose > 0 and a == 0):
					tqdm.write("\n\tbatch_aspAttn [Window Size: {}]: {}".format(self.args.ctx_win_size, attention_weights.size()))

			attention_weights = attention_weights.unsqueeze(2)
			batch_aspRep = batch_aspProjDoc * attention_weights.expand_as(batch_aspProjDoc)
			if(verbose > 0 and a == 0):
				tqdm.write("\n\tbatch_aspRep: {}".format( batch_aspRep.size() ))
			batch_aspRep = torch.sum(batch_aspRep, dim = 1)
			if(verbose > 0 and a == 0):
				tqdm.write("\tbatch_aspRep: {}".format( batch_aspRep.size() ))


			# Store the results (Attention & Representation) for this aspect
			lst_batch_aspAttn.append(torch.transpose(attention_weights, 1, 2))
			lst_batch_aspRep.append(torch.unsqueeze(batch_aspRep, 1))


		batch_aspAttn = torch.cat(lst_batch_aspAttn, dim = 1)
		batch_aspRep = torch.cat(lst_batch_aspRep, dim = 1)

		if(verbose > 0):
			tqdm.write("\n[Output] <All {} Aspects>".format( self.args.num_aspects ))
			tqdm.write("[Output] batch_aspAttn: {}".format( batch_aspAttn.size() ))
			tqdm.write("[Output] batch_aspRep: {}".format( batch_aspRep.size() ))
			tqdm.write("============================== ==================================== ==============================\n")

		return batch_aspAttn, batch_aspRep, id_vector