import torch
import torch.nn as nn

from .ARRR import ARRR
from .utilities import *

import numpy as np



class ModelZoo():

	def __init__(self, logger, args, timer):

		self.mdl = None
		self.logger = logger
		self.args = args
		self.timer = timer

		self.num_users, self.num_items = loadInfo(args)
		self.logger.log("\n[INFO] # of Users: {:,}, # of Items: {:,}".format( self.num_users, self.num_items ))


	def createAndInitModel(self):

		self.timer.startTimer("init")
		self.logger.log("\nCreating model (Selected Model: {})..".format( self.args.model ))

		'''
		Model Creation
		'''
		self.createModel()

		# Port to GPU, if necessary
		if(self.args.use_cuda):
			self.mdl.cuda()
			self.logger.log("[args.use_cuda: {}] Model is on the GPU! (args.gpu: {}, torch.cuda.current_device(): {})".format(
				self.args.use_cuda, self.args.gpu, torch.cuda.current_device() ))

		self.logger.log("Model created! {}".format( self.timer.getElapsedTimeStr("init", conv2Mins = True) ))


		'''
		Model Initialization
		'''
		self.initModel()


		# Cleanup
		if(self.args.use_cuda):
			torch.cuda.empty_cache()

		self.logger.log("\nInitialization Complete.. {}".format( self.timer.getElapsedTimeStr("init", conv2Mins = True) ))
		return self.mdl


	def createModel(self):

		# Update vocabulary size to include <pad> and <unk>
		# We use a vocabulary size of 50,000 in our experiments
		# However, that does not include <pad>, which is used to 'pad' all documents to the same length
		# Additionally, all OOV words are replaced with <unk>
		self.args.vocab_size = self.args.vocab_size + 2
		self.mdl = ARRR(self.logger, self.args, self.num_users, self.num_items)


	def initModel(self):

		self.initARRR()

	def initARRR(self):

		self.loadDocs()
		self.loadWordEmbeddings()

	# Load the user documents & item documents
	def loadDocs(self):

		uid_userDoc_path = "{}{}{}".format( self.args.input_dir, self.args.dataset, fp_uid_userDoc )
		iid_itemDoc_path = "{}{}{}".format( self.args.input_dir, self.args.dataset, fp_iid_itemDoc )

		# User Documents
		self.logger.log("\nLoading uid_userDoc from \"{}\"..".format( uid_userDoc_path ))
		np_uid_userDoc = np.load( uid_userDoc_path )

		self.mdl.uid_userDoc.weight.data.copy_(torch.from_numpy(np_uid_userDoc).long())
		self.logger.log("uid_userDoc loaded! [uid_userDoc: {}]".format( np_uid_userDoc.shape ))
		del np_uid_userDoc

		# Item Documents
		self.logger.log("\nLoading iid_itemDoc from \"{}\"..".format( iid_itemDoc_path ))
		np_iid_itemDoc = np.load( iid_itemDoc_path )

		self.mdl.iid_itemDoc.weight.data.copy_(torch.from_numpy(np_iid_itemDoc).long())
		self.logger.log("iid_itemDoc loaded! [iid_itemDoc: {}]".format( np_iid_itemDoc.shape ))
		del np_iid_itemDoc


	# Load/Randomly initialize the word embeddings
	def loadWordEmbeddings(self):

		if(self.args.pretrained_src == 1 or self.args.pretrained_src == 2):

			# Load pretrained word embeddings
			# 1: w2v (Google News), 300-dimensions
			# 2: GloVe (6B, 400K, 100d), this is included as it works better for D-Attn
			if(self.args.pretrained_src == 1):
				wid_wEmbed_path = "{}{}{}".format( self.args.input_dir, self.args.dataset, fp_wid_wordEmbed )
			elif(self.args.pretrained_src == 2):
				wid_wEmbed_path = "{}{}{}".format( self.args.input_dir, self.args.dataset, fp_wid_wordEmbed )

			self.logger.log("\nLoading pretrained word embeddings from \"{}\"..".format( wid_wEmbed_path ))
			np_wid_wEmbed = np.load( wid_wEmbed_path )

			print(len(np_wid_wEmbed))
			print(len(self.mdl.wid_wEmbed.weight))

			self.mdl.wid_wEmbed.weight.data.copy_(torch.from_numpy(np_wid_wEmbed))
			self.logger.log("Pretrained word embeddings loaded! [wid_wEmbed: {}]".format( np_wid_wEmbed.shape ))
			del np_wid_wEmbed

		else:

			# Randomly initialize word embeddings, using random uniform distribution from [rand_uniform_dist_min, rand_uniform_dist_max]
			rand_uniform_dist_min = -0.01
			rand_uniform_dist_max = 0.01
			self.mdl.wid_wEmbed.weight.data.uniform_(rand_uniform_dist_min, rand_uniform_dist_max)
			self.logger.log("\nWord embeddings are randomly initialized using random uniform distribution from [{:.2f}, {:.2f}]..".format(
				rand_uniform_dist_min, rand_uniform_dist_max ))

		# Ensures that the embeddings for <pad> and <unk> are always zero vectors
		self.mdl.wid_wEmbed.weight.data[PAD_idx].fill_(0)
		self.mdl.wid_wEmbed.weight.data[UNK_idx].fill_(0)


	'''
	Optimizer & Loss Function
	'''

	# Optimizer
	def selectOptimizer(self, optimizer = "Adam", learning_rate = 2E-3, L2_reg = 0):

		self.optimizer = optimizer.strip()

		# Set of parameters that need to be optimized, i.e. requires_grad == True
		if("ARRR" in self.args.model):
			opt_params = self.ARRR_Params()
		else:
			opt_params = filter(lambda p: p.requires_grad, self.mdl.parameters())

		if(self.optimizer == "Adam"):
			return optim.Adam(opt_params, lr = learning_rate)
		elif(self.optimizer == "RMSProp"):
			return optim.RMSprop(opt_params, lr = learning_rate)
		elif(self.optimizer == "SGD"):
			return optim.SGD(opt_params, lr = learning_rate)

		# Use Adam by default
		self.optimizer = "Adam"
		return optim.Adam(opt_params, lr = learning_rate)


	# (Optional) Apply a different LR to ARL parameters (Optional)
	# Apply L2 regularization to the User Bias & Item Bias
	def ARRR_Params(self):

		normalParams, paramsWithL2Reg = [], []
		lstL2RegParamNames = []

		for name, param in self.mdl.named_parameters():

			if(not param.requires_grad):
				continue

			# For AIE, L2 regularization is applied to the user & item bias
			if(self.args.L2_reg > 0.0 and ("uid_userOffset" in name or "iid_itemOffset" in name)):
				paramsWithL2Reg.append(param)
				lstL2RegParamNames.append(name)
				continue

			# All the other parameters, with default LR, and no L2 regularization
			normalParams.append(param)

		if(lstL2RegParamNames):
			self.logger.log("\nParameters with L2 Regularization (Regularization Strength: {}):\n{}".format(
				self.args.L2_reg, ", ".join(lstL2RegParamNames) ))

		return [{'params': paramsWithL2Reg, 'lr': self.args.learning_rate, 'weight_decay': self.args.L2_reg},
				{'params': normalParams, 'lr': self.args.learning_rate}]


	# Loss Function
	def selectLossFunction(self, loss_function = "MSELoss"):

		self.loss_function = loss_function.strip()

		if(self.loss_function == "MSELoss"):
			return nn.MSELoss()
		elif(self.loss_function == "L1Loss"):
			return nn.L1Loss()
		elif(self.loss_function == "SmoothL1Loss"):
			# return nn.SmoothL1Loss()
			return nn.SmoothL1Loss(beta=0.7)

		# Use MSELoss by default
		self.loss_function = "MSELoss"
		return nn.MSELoss()

