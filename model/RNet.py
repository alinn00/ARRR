import pickle
import torch
import torch.nn as nn
import numpy as np

from FILEPATHS import fp_split_train, fp_split_ntrain
from model.Mlp import MLP
from model.utilities import to_var


class RNet(nn.Module):
    def __init__(self, logger, args, num_users, num_items):
        super(RNet, self).__init__()
        self.logger = logger
        self.args = args
        self.num_users = num_users
        self.num_items = num_items

        self.device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    def forward(self, args, batch_uid, batch_iid):
        data_path = f"{args.input_dir}{args.dataset}{fp_split_train}"

        ratings_matrix = self.to_rating_matrix(data_path)

        user_ratings = ratings_matrix[batch_uid, :].float().to(self.device)
        user_ratings = self.normalize(user_ratings)
        user_mlp = MLP(self.num_items, self.args.h1).to(self.device)
        userlatent_vector = user_mlp(user_ratings)

        item_ratings = ratings_matrix[:, batch_iid].float().transpose(0, 1).to(self.device)
        item_ratings = self.normalize(item_ratings)
        item_mlp = MLP(self.num_users, self.args.h1).to(self.device)
        itemlatent_vector = item_mlp(item_ratings)

        return userlatent_vector, itemlatent_vector

    @staticmethod
    def normalize(tensor, eps=1e-6):
        norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
        return tensor / (norm + eps)

    def to_rating_matrix(self, pkl_file):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        rating_matrix = np.zeros((self.num_users, self.num_items))
        for uid, iid, rating in data:
            rating_matrix[uid, iid] = rating

        path = f"{self.args.input_dir}{self.args.dataset}{fp_split_ntrain}"

        np.save(path, rating_matrix)
        ratings_matrix = torch.from_numpy(np.load(path)).to(self.device)

        return ratings_matrix
