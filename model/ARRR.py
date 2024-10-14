import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.AspectImp import AspectImp
from model.AspectRep import AspectRep

from model.ANet import ANet
from model.RNet import RNet
from model.utilities import to_var


class ARRR(nn.Module):

    def __init__(self, logger, args, num_users, num_items):
        super(ARRR, self).__init__()

        self.logger = logger
        self.args = args

        self.num_users = num_users
        self.num_items = num_items

        # Dropout for the User & Item Aspect-Based Representations
        if self.args.dropout_rate > 0.0:
            self.userAspRepDropout = nn.Dropout(p=self.args.dropout_rate)
            self.itemAspRepDropout = nn.Dropout(p=self.args.dropout_rate)

        # Global Offset/Bias (Trainable)
        self.globalOffset = nn.Parameter(torch.Tensor(1), requires_grad=True)

        # User Offset/Bias & Item Offset/Bias
        self.uid_userOffset = nn.Embedding(self.num_users, 1)
        self.uid_userOffset.weight.requires_grad = True

        self.iid_itemOffset = nn.Embedding(self.num_items, 1)
        self.iid_itemOffset.weight.requires_grad = True

        # Initialize Global Bias with 0
        self.globalOffset.data.fill_(0)

        # Initialize All User/Item Offset/Bias with 0
        self.uid_userOffset.weight.data.fill_(0)
        self.iid_itemOffset.weight.data.fill_(0)

        self.shared_ANet = ANet(logger, args, num_users, num_items)
        self.shared_RNet = RNet(logger, args, num_users, num_items)

        self.uid_userDoc = nn.Embedding(self.num_users, self.args.max_doc_len)
        self.uid_userDoc.weight.requires_grad = False

        self.iid_itemDoc = nn.Embedding(self.num_items, self.args.max_doc_len)
        self.iid_itemDoc.weight.requires_grad = False

        # Word Embeddings (Input)
        self.wid_wEmbed = nn.Embedding(self.args.vocab_size, self.args.word_embed_dim)
        self.wid_wEmbed.weight.requires_grad = False

        # Aspect Representation Learning - Single Aspect-based Attention Network (Shared between User & Item)
        self.shared_AspectRep = AspectRep(logger, args, self.num_users, self.num_items)
        self.shared_AspectImp = AspectImp(logger, args)

        # W
        self.weight_matrix = nn.Parameter(torch.Tensor(self.args.h1 * 2 + self.args.d1, 1), requires_grad=True)
        self.weight_matrix.data.uniform_(-0.01, 0.01)

    def forward(self, args, batch_uid, batch_iid):

        # User & Item Bias
        batch_userOffset = self.uid_userOffset(batch_uid)
        batch_itemOffset = self.iid_itemOffset(batch_iid)

        # Input
        batch_userDoc = self.uid_userDoc(batch_uid)
        batch_itemDoc = self.iid_itemDoc(batch_iid)

        # Embedding Layer
        batch_userDocEmbed = self.wid_wEmbed(batch_userDoc.long())
        batch_itemDocEmbed = self.wid_wEmbed(batch_itemDoc.long())

        # ===================================================================== User Aspect-Based Representations =====================================================================
        # Aspect-based Representation Learning for User
        userAspAttn, userAspRep, uidvector = self.shared_AspectRep(args, batch_userDocEmbed, batch_uid, self.num_users)

        # ===================================================================== User Aspect-Based Representations =====================================================================

        # ===================================================================== Item Aspect-Based Representations =====================================================================
        # Aspect-based Representation Learning for Item

        itemAspAttn, itemAspRep, iidvector = self.shared_AspectRep(args, batch_itemDocEmbed, batch_iid, self.num_items)

        # ===================================================================== Item Aspect-Based Representations =====================================================================

        userAspImpt, itemAspImpt = self.shared_AspectImp(userAspRep, itemAspRep)


        if self.args.dropout_rate > 0.0:
            userAspRep = self.userAspRepDropout(userAspRep)
            itemAspRep = self.itemAspRepDropout(itemAspRep)

        userANetLF, itemANetLF = self.shared_ANet(userAspRep, itemAspRep, userAspImpt, itemAspImpt)

        user_latent_vector, item_latent_vector = self.shared_RNet(args, batch_uid, batch_iid)
        user_latent_vector = user_latent_vector.cuda()
        item_latent_vector = item_latent_vector.cuda()

        final_user_embedding = torch.cat((user_latent_vector, userANetLF), dim=1)
        final_item_embedding = torch.cat((item_latent_vector, itemANetLF), dim=1)

        final_user_embedding = torch.cat((uidvector, final_user_embedding), dim=1)
        final_item_embedding = torch.cat((iidvector, final_item_embedding), dim=1)

        raw_prediction_scores = torch.matmul((final_user_embedding * final_item_embedding), self.weight_matrix)

        prediction_scores = raw_prediction_scores + batch_userOffset + batch_itemOffset

        # Include Global Bias
        prediction_scores = prediction_scores + self.globalOffset
        # print(raw_prediction_scores)

        # print(f"prediction_scores ： {prediction_scores}")

        return prediction_scores
 