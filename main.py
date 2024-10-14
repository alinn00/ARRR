import torch
from model.utilities import *
from model.ModelZoo import ModelZoo
from model.Logger import Logger
from model.Timer import Timer

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import argparse
from datetime import datetime

def get_Recall(t, r):
	return len(np.intersect1d(t, r)) / len(t)


def get_Precision(t, r, top_k):
	return len(np.intersect1d(t, r)) / top_k


def get_NDCG(t, r, top_k):
	idcg = 0
	for i in range(top_k):
		idcg += 1 / np.log2(i + 2)  # i start from 0, so need add 2 instead.
	dcg = 0
	for index, item in enumerate(r):
		if item in t:
			dcg += 1 / np.log2(index + 2)
	return dcg / idcg


def evaluate_topk(device, model, top_k, train_data_path, eval_data_path, test_data_path):
	with open(train_data_path, 'rb') as f:
		train_data = list(pickle.load(f))
	with open(test_data_path, 'rb') as f:
		test_data = list(pickle.load(f))
	with open(eval_data_path, 'rb') as f:
		test_data.extend(list(pickle.load(f)))

	train_data = pd.DataFrame(train_data, columns=['user', 'item', 'rating'])
	test_data = pd.DataFrame(test_data, columns=['user', 'item', 'rating'])
	# train_data.to_csv('./tmp_train.csv', index=False)
	# test_data.to_csv('./tmp_test.csv', index=False)
	user_id_max = max(train_data['user'].max(), test_data['user'].max()) + 1
	item_id_max = max(train_data['item'].max(), test_data['item'].max()) + 1
	# item_list = list(range(0, item_id_max))  # [0, 1, 2, ..., item_id_max]
	# remove the items which item's interaction number is lower than 5
	item_counts = train_data['item'].value_counts()
	item_list_filtered = item_counts[item_counts > 20].index.tolist()
	print(f"Before filtering, item number is: {item_id_max}")
	print(f"After filtering, item number is: {len(item_list_filtered)}")

	# print(f"user_id_max = {user_id_max}")
	# print(f"item_id_max = {item_id_max}")

	test_users = test_data['user'].unique()
	# test_batch_size = 64  # the user numbers in a batch
	# num_user_batchs = len(test_users) // test_batch_size + 1
	train_user_items = train_data.groupby('user')['item'].apply(list).to_dict()

	items_for_user = {}
	for user in test_users:
		items_for_user[user] = [i for i in item_list_filtered if i not in train_user_items[user]]

	Recall, Precision, NDCG = [], [], []
	with torch.no_grad():
		for user in tqdm(test_users):
			model.eval()
			# batch_uid = torch.tensor([user] * len(item_list_filtered))
			# batch_iid = torch.tensor(item_list_filtered)
			batch_uid = torch.tensor([user] * len(items_for_user[user]))
			batch_iid = torch.tensor(items_for_user[user])
			# print(f"user_tensor: {batch_uid}")
			# print(f"item_tensor: {batch_iid}")
			batch_uid = to_var(batch_uid, use_cuda=True, phase="Test")
			batch_iid = to_var(batch_iid, use_cuda=True, phase="Test")
			rating_pred = torch.squeeze(model(args, batch_uid, batch_iid)).cpu()

			# pair: (batch_uid[i], batch_iid[i], rating_pred[i])

			# print(f"rating_pred: {rating_pred}")
			# get top-k recommendation lists
			_, top_k_indices_sorted = torch.topk(rating_pred, k=10, dim=0)

			true_list = test_data.loc[test_data['user'] == user, 'item'].values.reshape(-1)
			pred_list = batch_iid[top_k_indices_sorted].cpu().numpy()

			# print(f"pred_list : {pred_list}")
			# print(f"true_list : {true_list}")

			Recall.append(get_Recall(true_list, pred_list))
			Precision.append(get_Precision(true_list, pred_list, top_k))
			NDCG.append(get_NDCG(true_list, pred_list, top_k))

	recall = np.mean(Recall)
	precision = np.mean(Precision)
	f1_score = 0
	if recall + precision > 1e-6:
		f1_score = 2 * recall * precision / (recall + precision)
	ndcg = np.mean(NDCG)

	return recall, precision, f1_score, ndcg


def evaluate(model, set_loader, epoch_num=-1, use_cuda=True, phase="Dev", print_txt=True):
	all_rating_true = []
	all_rating_pred = []

	for batch_num, (batch_uid, batch_iid, batch_rating) in enumerate(set_loader):
		# pair (uid, iid, rating), Amonologue
		model.eval()

		# print(f"batch_num: {batch_num}")
		# print(f"batch_uid: {batch_uid}")
		# print(f"batch_iid: {batch_iid}")
		# print(f"batch_rating: {batch_rating}")
		# print(f"type(batch_uid): {type(batch_uid)}")
		# print(f"type(batch_iid): {type(batch_iid)}")

		batch_uid = to_var(batch_uid, use_cuda=use_cuda, phase=phase)
		batch_iid = to_var(batch_iid, use_cuda=use_cuda, phase=phase)

		rating_pred = torch.squeeze(model(args, batch_uid, batch_iid))
		rating_pred = rating_pred.cpu()

		# print("batch_rating: ", batch_rating)
		# print("rating_pred: ",rating_pred.data)

		all_rating_true.extend(batch_rating)
		all_rating_pred.extend(rating_pred.data)
	# print(all_rating_true)
	# print(f"batch size : {batch_rating.size()}")
	# print(f"number of users in a batch: {batch_rating.size()}")
	# _, ground_truth_indices = torch.topk(batch_rating, k=10)
	# _, prediction_indices = torch.topk(rating_pred.data, k=10)
	# print(f"ground_truth_indices: {ground_truth_indices}")
	# print(f"prediction_indices: {prediction_indices}")

	# print(all_rating_true)

	MSE = mean_squared_error(all_rating_true, all_rating_pred)
	MAE = mean_absolute_error(all_rating_true, all_rating_pred)
	RMSE = np.sqrt(MSE)

	logger.log("[{}] {:6s} MSE: {:.5f}, MAE: {:.5f}, RMSE: {:.5f}".format(
		"Epoch {:d}".format(epoch_num + 1) if epoch_num >= 0 else "Initial",
		"[{}]".format(phase), MSE, MAE, RMSE), print_txt=print_txt)
	# logger.log("[{}] {:6s} MSE: {:.5f}, MAE: {:.5f}".format( "Epoch {:d}".format( epoch_num + 1 ) if epoch_num >= 0 else "Initial",
	# 	"[{}]".format( phase ), MSE, MAE), print_txt = print_txt)

	return MSE, MAE, RMSE


def main():
	parser = argparse.ArgumentParser()

	# Dataset & Model
	parser.add_argument("-d", dest="dataset", type=str, default="musical_instruments_5",
						help="Dataset for Running Experiments")
	parser.add_argument("-m", dest="model", type=str, default="ARRR", help="Model Name")

	# General Hyperparameters
	parser.add_argument("-bs", dest="batch_size", type=int, default=128, help="Batch Size")
	parser.add_argument("-e", dest="epochs", type=int, default=10, help="Number of Training Epochs")
	parser.add_argument("-lr", dest="learning_rate", type=float, default=2E-3,
						help="Learning Rate (Default: 0.002, i.e 2E-3)")
	parser.add_argument("-opt", dest="optimizer", type=str, default="Adam",
						help="Optimizer, e.g. Adam|RMSProp|SGD")
	parser.add_argument("-loss_func", dest="loss_function", type=str, default="SmoothL1Loss",
						help="Loss Function, e.g. MSELoss|L1Loss|SmoothL1Loss")
	parser.add_argument("-dr", dest="dropout_rate", type=float, default=0.5, help="Dropout rate")

	parser.add_argument("-MDL", dest="max_doc_len", type=int, default=500,
						help="Maximum User/Item Document Length")
	parser.add_argument("-v", dest="vocab_size", type=int, default=15952, help="Vocabulary Size")

	parser.add_argument("-WED", dest="word_embed_dim", type=int, default=300,
						help="Number of Dimensions for the Word Embeddings")
	parser.add_argument("-p", dest="pretrained_src", type=int, default=1,
						help="Source of Pretrained Word Embeddings? 0: Randomly Initialized (Random Uniform Dist. from [-0.01, 0.01]), 1: w2v (Google News, 300d), 2: GloVe (6B, 400K, 100d) (Default: 1)")

	# ARRR Hyperparameters
	parser.add_argument("-K", dest="num_aspects", type=int, default=5, help="Number of Aspects ")
	parser.add_argument("-h1", dest="h1", type=int, default=30,
						help="Dimensionality of the Aspect-level Representations")
	parser.add_argument("-c", dest="ctx_win_size", type=int, default=5,
						help="Window Size (i.e. Number of Words) for Calculating Attention")
	parser.add_argument("-h2", dest="h2", type=int, default=50,
						help="Dimensionality of the Hidden Layers used for Aspect Importance Estimation")
	parser.add_argument("-L2_reg", dest="L2_reg", type=float, default=1E-6,
						help="L2 Regularization for User & Item Bias")
	parser.add_argument("-d1", dest="d1", type=int, default=20, help="ID_EMBEDDING_DIM")

	parser.add_argument("-rs", dest="random_seed", type=int, default=1337, help="Random Seed")
	parser.add_argument("-dc", dest="disable_cuda", type=int, default=0,
						help="Disable CUDA? (Default: 0, i.e. run using GPU (if available))")
	parser.add_argument("-gpu", dest="gpu", type=int, default=0, help="Which GPU to use?")
	parser.add_argument("-vb", dest="verbose", type=int, default=0,
						help="Show debugging/miscellaneous information?")
	parser.add_argument("-die", dest="disable_initial_eval", type=int, default=0,
						help="Disable initial Dev/Test evaluation?")

	global args
	args = parser.parse_args()

	args.use_cuda = not args.disable_cuda and torch.cuda.is_available()
	del args.disable_cuda

	# Initial Setup
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)

	if (args.use_cuda):
		select_gpu(args.gpu)
		torch.cuda.set_device(args.gpu)
		torch.backends.cudnn.enabled = True
		torch.backends.cudnn.deterministic = True
		torch.cuda.manual_seed(args.random_seed)
	else:
		print("\n[args.use_cuda: {}] The program will be executed on the CPU!!".format(args.use_cuda))

	# Timer & Logging
	timer = Timer()
	timer.startTimer()

	uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	args.input_dir = "./datasets/{}/".format(args.dataset)
	args.out_dir = "./experimental_results/{} - {}/".format(args.dataset, args.model)
	log_path = "{}{}-{}".format(args.out_dir, uuid, 'logs.txt')
	global logger  # Amonologue
	logger = Logger(args.out_dir, log_path, args)

	# Create model
	modelZoo = ModelZoo(logger, args, timer)
	model = modelZoo.createAndInitModel()

	# Load training/validation/testing sets
	train_set, train_loader, dev_set, dev_loader, test_set, test_loader = loadTrainDevTest(logger, args)

	logger.log("Train/Dev/Test splits loaded! {}".format(timer.getElapsedTimeStr("init", conv2Mins=True)))

	# Initial Evaluation - Validation & Testing
	if (not args.disable_initial_eval):
		logger.log("\nPerforming initial evaluation for VALIDATION set..")
		evaluate(model, dev_loader, use_cuda=args.use_cuda, phase="Dev")
		logger.log("\nPerforming initial evaluation for TESTING set..")
		evaluate(model, test_loader, use_cuda=args.use_cuda, phase="Test")
		logger.log("\nInitial Evaluation Complete.. {}".format(timer.getElapsedTimeStr("init", conv2Mins=True)))

	# Loss Function, Custom Regularizers, Optimizer
	criterion = modelZoo.selectLossFunction(loss_function=args.loss_function)
	opt = modelZoo.selectOptimizer(optimizer=args.optimizer, learning_rate=args.learning_rate, L2_reg=args.L2_reg)
	logger.log("\nOptimizer: {}, Loss Function: {}".format(modelZoo.optimizer, modelZoo.loss_function))

	# Model Information
	generate_mdl_summary(model, logger)

	lstTrainingLoss = []
	lstDevMSE = []
	lstDevMAE = []
	lstDevRMSE = []
	lstTestMSE = []
	lstTestMAE = []
	lstTestRMSE = []

	# if os.path.exists(model_path):
	timer.startTimer("training")
	for epoch_num in range(args.epochs):

		# Training loop, using mini-batches
		print("\n")
		losses = []
		for batch_num, (batch_uid, batch_iid, batch_rating) in enumerate(
				tqdm(train_loader, "Epoch {:d}".format(epoch_num + 1))):
			# Set to training mode, zero out the gradients
			model.train()
			opt.zero_grad()  # 将优化器中的梯度归零

			batch_uid = to_var(batch_uid, use_cuda=args.use_cuda)
			batch_iid = to_var(batch_iid, use_cuda=args.use_cuda)
			rating_true = to_var(batch_rating, use_cuda=args.use_cuda)

			rating_pred = torch.squeeze(model(args, batch_uid, batch_iid))
			# print(f"rating_pred : {rating_pred.float()}")
			# print(f"rating_true : {rating_true.float()}")
			loss = criterion(rating_pred.float(), rating_true.float())

			loss.backward()
			opt.step()

			# losses.append(loss.data[0])
			losses.append(loss.item())

		trainingLoss = np.mean(losses)
		lstTrainingLoss.append(trainingLoss)
		logger.log("\n[Epoch {:d}/{:d}] Training Loss: {:.5f}\t{}".format(epoch_num + 1, args.epochs, trainingLoss,
																		  timer.getElapsedTimeStr("training",
																								  conv2HrsMins=True)))

		# Evaluation - Validation & Testing
		devMSE, devMAE, devRMSE = evaluate(model, dev_loader, epoch_num=epoch_num, use_cuda=args.use_cuda,
										   phase="Dev")
		# testMSE, testMAE = evaluate(mdl, 3,test_loader, epoch_num = epoch_num, use_cuda = args.use_cuda, phase = "Test")
		testMSE, testMAE, testRMSE = evaluate(model, test_loader, epoch_num=epoch_num, use_cuda=args.use_cuda,
											  phase="Test")

		lstDevMSE.append(devMSE)
		lstDevMAE.append(devMAE)
		lstDevRMSE.append(devRMSE)
		lstTestMSE.append(testMSE)
		lstTestMAE.append(testMAE)
		lstTestRMSE.append(testRMSE)

	# test the performance of top-k recommendaton
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	top_k = 10
	# k = 3 # this is not top_k
	# data path
	train_data_path = "{}{}{}".format(args.input_dir, args.dataset, fp_split_train)
	eval_data_path = "{}{}{}".format(args.input_dir, args.dataset, fp_split_dev)
	test_data_path = "{}{}{}".format(args.input_dir, args.dataset, fp_split_test)
	# evaluate_topk(device, model, k, train_data_path, eval_data_path, test_data_path)
	recall, precision, f1_score, ndcg = evaluate_topk(device, model, top_k, train_data_path, eval_data_path,
													  test_data_path)
	#print(f"Recall@{top_k} = {recall}, Precision@{top_k} = {precision}, F1-score@{top_k} = {f1_score}, NDCG@{top_k} = {ndcg}")
	# ----------------------------------------------------------------------------------------------------------------------
	logger.log(
		f"Recall@{top_k} = {recall}, Precision@{top_k} = {precision}, F1-score@{top_k} = {f1_score}, NDCG@{top_k} = {ndcg}")

	logger.log("\n[Training Loss]\n{}".format([float("{:.5f}".format(i)) for i in lstTrainingLoss]))
	logger.log("\n[Dev MSE]\n{}".format([float("{:.5f}".format(i)) for i in lstDevMSE]))
	logger.log("\n[Dev MAE]\n{}".format([float("{:.5f}".format(i)) for i in lstDevMAE]))
	logger.log("\n[Dev RMSE]\n{}".format([float("{:.5f}".format(i)) for i in lstDevRMSE]))
	logger.log("[Test MSE]\n{}".format([float("{:.5f}".format(i)) for i in lstTestMSE]))
	logger.log("[Test MAE]\n{}\n".format([float("{:.5f}".format(i)) for i in lstTestMAE]))
	logger.log("[Test RMSE]\n{}\n".format([float("{:.5f}".format(i)) for i in lstTestRMSE]))

	epoch_num_forBestDevMSE, bestDevMSE, testMSE_forBestDevMSE = getBestPerf(lstDevMSE, lstTestMSE)
	logger.log("\nBest Dev MSE: {:.5f} (Obtained during Evaluation #{:d})".format(bestDevMSE, epoch_num_forBestDevMSE))
	logger.log("\nTest MSE: {:.5f}".format(testMSE_forBestDevMSE))

	epoch_num_forBestDevMAE, bestDevMAE, testMAE_forBestDevMAE = getBestPerf(lstDevMAE, lstTestMAE)
	logger.log("\nBest Dev MAE: {:.5f} (Obtained during Evaluation #{:d})".format(bestDevMAE, epoch_num_forBestDevMAE))
	logger.log("\nTest MAE: {:.5f}".format(testMAE_forBestDevMAE))

	epoch_num_forBestDevRMSE, bestDevRMSE, testRMSE_forBestDevRMSE = getBestPerf(lstDevRMSE, lstTestRMSE)
	logger.log(
		"\nBest Dev RMSE: {:.5f} (Obtained during Evaluation #{:d})".format(bestDevRMSE, epoch_num_forBestDevRMSE))
	logger.log("\nTest RMSE: {:.5f}".format(testRMSE_forBestDevRMSE))

	logger.log("\nEnd of Program! {}".format(timer.getElapsedTimeStr(conv2HrsMins=True)))
	print("\n\n\n")


if __name__ == "__main__":
	main()
