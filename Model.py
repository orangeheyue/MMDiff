import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
import math
from Utils.Utils import *

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class GCNModel(nn.Module):
	'''
		Multi-View Diffusion for multimodal recommender system.


	'''
	def __init__(self, image_embedding, text_embedding, audio_embedding=None, modal_fusion=True):
		super(GCNModel, self).__init__()
		
		self.sparse = True
		self.gcn_layer_num = 2
		self.edgeDropper = SpAdjDropEdge(args.keepRate)
		self.reg_weight = 1e-5
		self.batch_size = 1024
		self.modal_fusion = modal_fusion


		# modal feature embedding
		self.image_embedding = image_embedding
		self.text_embedding = text_embedding
		self.audio_embedding = audio_embedding

		# user & item embdding
		self.user_embedding = nn.Embedding(args.user, args.latdim)    # self.user_embedding .shape: torch.Size([9308, 64]) self.item_id_embedding.shape: torch.Size([6710, 64])
		self.item_id_embedding = nn.Embedding(args.item, args.latdim)
		nn.init.xavier_uniform_(self.user_embedding.weight)
		nn.init.xavier_uniform_(self.item_id_embedding.weight)

		# modal feature projection
		if self.image_embedding is not None:
			self.image_residual_project = nn.Sequential(
				nn.Linear(in_features=self.image_embedding.shape[1], out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU()
			)
			self.image_modal_project = nn.Sequential(
				nn.Linear(in_features=args.latdim, out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU()
			)

		if self.text_embedding is not None:
			self.text_residual_project = nn.Sequential(
				nn.Linear(in_features=self.text_embedding.shape[1], out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU()
			)
			self.text_modal_project = nn.Sequential(
				nn.Linear(in_features=args.latdim, out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU()
			)

		if self.audio_embedding is not None:
			self.audio_residual_project = nn.Sequential(
				nn.Linear(in_features=self.audio_embedding.shape[1], out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU()
			)
			self.audio_modal_project = nn.Sequential(
				nn.Linear(in_features=args.latdim, out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU()
			)

		self.softmax = nn.Softmax(dim=-1)

		self.gate_image_modal = nn.Sequential(
            nn.Linear(args.latdim, args.latdim),
            nn.Sigmoid()
        )

		self.gate_text_modal = nn.Sequential(
            nn.Linear(args.latdim, args.latdim),
            nn.Sigmoid()
        )

		self.gate_audio_modal = nn.Sequential(
            nn.Linear(args.latdim, args.latdim),
            nn.Sigmoid()
        )


		self.caculate_common = nn.Sequential(
            nn.Linear(args.latdim, args.latdim),
            nn.LeakyReLU(),
            nn.Linear(args.latdim, 1, bias=False)
        )

		self.init_modal_weight()

	def init_modal_weight(self):
		"""
		初始化模型权重
		"""
		# 对图像模态投影层中的线性层权重进行初始化（如果有图像模态嵌入的话）
		if self.image_embedding is not None:
			for layer in self.image_modal_project:
				if isinstance(layer, nn.Linear):
					nn.init.xavier_uniform_(layer.weight)
		# 对文本模态投影层中的线性层权重进行初始化（如果有文本模态嵌入的话）
		if self.text_embedding is not None:
			for layer in self.text_modal_project:
				if isinstance(layer, nn.Linear):
					nn.init.xavier_uniform_(layer.weight)
		# 对音频模态投影层中的线性层权重进行初始化（如果有音频模态嵌入的话）
		if self.audio_embedding is not None:
			for layer in self.audio_modal_project:
				if isinstance(layer, nn.Linear):
					nn.init.xavier_uniform_(layer.weight)
		
		# 对计算公共表示相关的线性层权重进行初始化
		for layer in self.caculate_common:
			if isinstance(layer, nn.Linear):
				nn.init.xavier_uniform_(layer.weight)
		
		# 对各模态的门控相关的线性层权重进行初始化
		for layer in self.gate_image_modal:
			if isinstance(layer, nn.Linear):
				nn.init.xavier_uniform_(layer.weight)
		for layer in self.gate_text_modal:
			if isinstance(layer, nn.Linear):
				nn.init.xavier_uniform_(layer.weight)
		for layer in self.gate_audio_modal:
			if isinstance(layer, nn.Linear):
				nn.init.xavier_uniform_(layer.weight)

		# 残差模块
		for layer in  self.image_residual_project:
				if isinstance(layer, nn.Linear):
					nn.init.xavier_uniform_(layer.weight)
		# 残差模块
		for layer in  self.text_residual_project:
				if isinstance(layer, nn.Linear):
					nn.init.xavier_uniform_(layer.weight)
		# 残差模块
		if self.audio_embedding is not None:
			for layer in  self.audio_residual_project:
					if isinstance(layer, nn.Linear):
						nn.init.xavier_uniform_(layer.weight)

		
	def getItemEmbeds(self):
		'''
			获取Item embedding
		'''
		return self.item_id_embedding.weight

	def getUserEmbeds(self):
		'''
			获取User embedding
		'''
		return self.user_embedding.weight
	
	def getImageFeats(self):
		'''
			获取图像模态特征
		'''
		if self.image_embedding is not None:
			x = self.image_residual_project(self.image_embedding)
			image_modal_feature = self.image_modal_project(x)
			image_modal_feature += x
		return image_modal_feature

	def getTextFeats(self):
		'''
			获取文本模态特征
		'''
		if self.text_embedding is not None:
			x = self.text_residual_project(self.text_embedding)
			text_modal_feature = self.text_modal_project(x)
			text_modal_feature += x
		return text_modal_feature
	
	def getAudioFeats(self):
		'''
			获取音频模态特征
		'''
		if self.audio_embedding is not None:
			x = self.audio_residual_project(self.audio_embedding)
			audio_modal_feature = self.audio_modal_project(x)
			audio_modal_feature += x
		return audio_modal_feature
	

	def multimodal_feature_fusion_adj(self, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj):
		'''
			多模态特征提取与fusion
		'''
		multimodal_feature_fusion_adj =  diffusion_ii_image_adj + diffusion_ii_text_adj + diffusion_ii_audio_adj

		return multimodal_feature_fusion_adj

	def user_item_GCN(self, original_ui_adj, diffusion_ui_adj):
		'''
			User-Item GCN
			original_ui_adj:size=(16018, 16018)
			diffusion_ui_adj:size=(6710, 6710)
			TODO: diffusion_ui_adj目前简单的进行融合

			original_ui_adj: tensor(indices=tensor([[    0, 10193, 10695,  ..., 16015, 16016, 16017],
								[    0,     0,     0,  ..., 16015, 16016, 16017]]),
				values=tensor([0.2500, 0.1443, 0.0606,  ..., 1.0000, 1.0000, 1.0000]),
				device='cuda:0', size=(16018, 16018), nnz=135100, layout=torch.sparse_coo) 
			
			diffusion_ui_adj: tensor(indices=tensor([[15828,     1,     2,  ..., 16012, 16013, 16016],
								[    0,     1,     2,  ..., 16012, 16013, 16016]]),
				values=tensor([0.0148, 1.0000, 1.0000,  ..., 2.0000, 2.0000, 2.0000]),
				device='cuda:0', size=(16018, 16018), nnz=51910, layout=torch.sparse_coo)

			adj: tensor(indices=tensor([[    0, 10193, 10695,  ..., 16012, 16013, 16016],
								[    0,     0,     0,  ..., 16012, 16013, 16016]]),
				values=tensor([0.2500, 0.1443, 0.0606,  ..., 2.0000, 2.0000, 2.0000]),
				device='cuda:0', size=(16018, 16018), nnz=187010, layout=torch.sparse_coo)
		'''
		#print("original_ui_adj:", original_ui_adj, "diffusion_ui_adj:", diffusion_ui_adj)
		adj = original_ui_adj + diffusion_ui_adj  #
		#print("adj:", adj)
		# adj = original_ui_adj  # 

		# adj1 = original_ui_adj
		# adj2 = diffusion_ui_adj 

		cat_embedding = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)

		all_embeddings = [cat_embedding]
		for i in range(self.gcn_layer_num):
			# temp_embeddings1 = torch.sparse.mm(adj1, cat_embedding)
			# cat_embedding = temp_embeddings1
			# all_embeddings += [cat_embedding]

			temp_embeddings2 = torch.sparse.mm(adj, cat_embedding)
			cat_embedding = temp_embeddings2
			all_embeddings += [cat_embedding]
		
		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
		content_embedding = all_embeddings
		#print("content_embedding:", content_embedding)

		return content_embedding


	def item_item_GCN(self,R, original_ui_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj=None):
		'''
			Item-Item GCN
		'''
		# 获取 物品+ID 特征
		image_modal_feature = self.getImageFeats()
		image_item_id_embedding =  torch.multiply(self.item_id_embedding.weight, self.gate_image_modal(image_modal_feature))

		text_modal_feature = self.getTextFeats()
		text_item_id_embedding = torch.multiply(self.item_id_embedding.weight, self.gate_text_modal(text_modal_feature))

		if args.data == 'tiktok':
			audio_modal_feature = self.getAudioFeats()
			audio_item_id_embedding = torch.multiply(self.item_id_embedding.weight, self.gate_audio_modal(audio_modal_feature))

		# print("original_ui_adj.shape:", original_ui_adj.shape)
		# user-user adj
		self.R = R
		# print("original_ui_adj:", original_ui_adj)
		# print("R:", R)
		if self.sparse:
			for _ in range(self.gcn_layer_num):
				image_item_id_embedding = torch.sparse.mm(diffusion_ii_image_adj, image_item_id_embedding)
		else:
			for _ in range(self.gcn_layer_num):
				image_item_id_embedding = torch.mm(diffusion_ii_image_adj, image_item_id_embedding)

		image_user_embedding = torch.sparse.mm(self.R, image_item_id_embedding) 
		image_ui_embedding = torch.cat([image_user_embedding, image_item_id_embedding], dim=0)

		if self.sparse:
			for _ in range(self.gcn_layer_num):
				text_item_id_embedding = torch.sparse.mm(diffusion_ii_text_adj, text_item_id_embedding)
		else:
			for _ in range(self.gcn_layer_num):
				text_item_id_embedding = torch.mm(diffusion_ii_text_adj, text_item_id_embedding)
		text_user_embedding = torch.sparse.mm(self.R, text_item_id_embedding) 
		text_ui_embedding = torch.cat([text_user_embedding, text_item_id_embedding], dim=0)
		
		if args.data == 'tiktok':

					if self.sparse:

						for _ in range(self.gcn_layer_num):
							audio_item_id_embedding = torch.sparse.mm(diffusion_ii_audio_adj, audio_item_id_embedding)
					else:
						for _ in range(self.gcn_layer_num):
							audio_item_id_embedding = torch.mm(diffusion_ii_audio_adj, audio_item_id_embedding)

					audio_user_embedding = torch.sparse.mm(self.R, audio_item_id_embedding) 
					audio_ui_embedding = torch.cat([audio_user_embedding, audio_item_id_embedding], dim=0)


		return (image_ui_embedding, text_ui_embedding, audio_ui_embedding) if args.data == 'tiktok' else (image_ui_embedding, text_ui_embedding)


	def gate_attention_fusion(self, image_ui_embedding, text_ui_embedding, audio_ui_embedding=None):
		'''
			GAT Attention Fusion
		'''
		if args.data == 'tiktok':

			attention_common = torch.cat([self.caculate_common(image_ui_embedding), self.caculate_common(text_ui_embedding), self.caculate_common(audio_ui_embedding)], dim=-1)
			weight_common = self.softmax(attention_common)
			common_embedding = weight_common[:, 0].unsqueeze(dim=1) * image_ui_embedding + weight_common[:, 1].unsqueeze(dim=1) * text_ui_embedding + weight_common[:, 2].unsqueeze(dim=1) * audio_ui_embedding
			sepcial_image_ui_embedding = image_ui_embedding - common_embedding
			special_text_ui_embedding  = text_ui_embedding - common_embedding
			special_audio_ui_embedding = audio_ui_embedding - common_embedding

			return sepcial_image_ui_embedding, special_text_ui_embedding, special_audio_ui_embedding, common_embedding
		else:
			attention_common = torch.cat([self.caculate_common(image_ui_embedding), self.caculate_common(text_ui_embedding)], dim=-1)
			weight_common = self.softmax(attention_common)
			common_embedding = weight_common[:, 0].unsqueeze(dim=1) * image_ui_embedding + weight_common[:, 1].unsqueeze(dim=1) * text_ui_embedding 
			sepcial_image_ui_embedding = image_ui_embedding - common_embedding
			special_text_ui_embedding  = text_ui_embedding - common_embedding

			return sepcial_image_ui_embedding, special_text_ui_embedding, common_embedding


	def bpr_loss(self, anc_embeds, pos_embeds, neg_embeds):
		"""
		BPR loss计算函数:
		Args:
			anc_embeds: 用户嵌入向量，形状应为 [batch_size, embed_dim]
			pos_embeds: 正样本嵌入向量，形状应为 [batch_size, embed_dim]
			neg_embeds: 负样本嵌入向量，形状应为 [batch_size, embed_dim]
		Returns:
			bpr_loss: 计算得到的BPR损失值(一个标量)
		"""
		# 简单检查输入张量维度是否符合预期（这里只是简单示意，可根据实际情况完善）
		assert anc_embeds.dim() == 2, "用户嵌入向量维度应为2"
		assert pos_embeds.dim() == 2, "正样本嵌入向量维度应为2"
		assert neg_embeds.dim() == 2, "负样本嵌入向量维度应为2"
		assert anc_embeds.shape == pos_embeds.shape, "用户嵌入与正样本嵌入维度应匹配"
		assert anc_embeds.shape == neg_embeds.shape, "用户嵌入与负样本嵌入维度应匹配"

		# 计算正样本和负样本的得分：
		pos_scores = torch.sum(torch.mul(anc_embeds, pos_embeds), dim=-1)  # 计算用户与正样本物品的点积之和，作为正样本得分，形状为 [batch_size]
		neg_scores = torch.sum(torch.mul(anc_embeds, neg_embeds), dim=-1)  # 计算用户与负样本物品的点积之和，作为负样本得分，形状为 [batch_size]

		# 计算BPR损失
		diff_scores = pos_scores - neg_scores
		bpr_loss = -1 * torch.mean(F.logsigmoid(diff_scores))  # 计算正样本得分与负样本得分之差的logsigmoid，取反并计算平均值，促使正样本得分高于负样本得分，作为排序损失（一个数值）

		# 计算正则化损失
		regularizer = 1.0 / 2 * (anc_embeds ** 2).sum() + 1.0 / 2 * (pos_embeds ** 2).sum() + 1.0 / 2 * (neg_embeds ** 2).sum()
		regularizer = regularizer / args.batch
		emb_loss = self.reg_weight * regularizer 

		# 正则化损失2
		reg_loss = self.reg_loss() * args.reg
 
		return bpr_loss, emb_loss, reg_loss
	
	
	def infoNCE_loss(self, view1, view2,  temperature):
		'''
			InfoNCE loss
		'''
		view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
		pos_score = torch.sum((view1 * view2), dim=-1)
		pos_score = torch.exp(pos_score / temperature)

		neg_score = (view1 @ view2.T) / temperature
		neg_score = torch.exp(neg_score).sum(dim=1)
		contrast_loss = -1 * torch.log(pos_score / neg_score).mean()

		return contrast_loss

	def reg_loss(self):
		ret = 0
		ret += self.user_embedding.weight.norm(2).square()
		ret += self.item_id_embedding.weight.norm(2).square()
		return ret 


	def forward(self, R, original_ui_adj, diffusion_ui_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj=None, diffusion_modal_fusion_ii_matrix=None):
		'''
			GCN 前向过程:
				1. 多模态特征提取与fusion
				2. User-Item GCN
				3. Item-Item GCN
				4. GAT Attention Fusion
				5. Contrastive: BPR loss + InfoNCE
			
			Args:
				original_ui_adj: 原始的user-item graph
				diffusion_ui_adj: 扩散模型生成的user-item graph
				diffusion_ii_image_adj: 扩散模型生成的item-item image modal graph
				diffusion_ii_text_adj: 扩散模型生成的item-item  text modal graph
				diffusion_ii_audio_adj: 扩散模型生成的item-item audi modal graph

			Return:
				User Embedding, Item Embedding

				original_ui_adj: tensor(indices=tensor([[    0, 10193, 10695,  ..., 16015, 16016, 16017],
							[    0,     0,     0,  ..., 16015, 16016, 16017]]),
							values=tensor([0.2500, 0.1443, 0.0606,  ..., 1.0000, 1.0000, 1.0000]),
							device='cuda:0', size=(16018, 16018), nnz=135100, layout=torch.sparse_coo)

				diffusion_ui_adj: tensor(indices=tensor([[10272,     1,     2,  ..., 16012, 16013, 16016],
									[    0,     1,     2,  ..., 16012, 16013, 16016]]),
					values=tensor([0.0154, 1.0000, 1.0000,  ..., 2.0000, 2.0000, 2.0000]),
					device='cuda:0', size=(16018, 16018), nnz=51910, layout=torch.sparse_coo)
		'''
		# 多模态特征提取与fusion
		# print("original_ui_adj:", original_ui_adj)
		# print("diffusion_ui_adj:", diffusion_ui_adj)
		content_embedding = self.user_item_GCN(original_ui_adj, diffusion_ui_adj)
		#print("user-item gcn-------->content_embedding.shape", content_embedding.shape) # torch.Size([16018, 64])

		if args.data == 'tiktok':
			
			if self.modal_fusion == True:
				diffusion_ii_image_adj += diffusion_modal_fusion_ii_matrix
				diffusion_ii_text_adj += diffusion_modal_fusion_ii_matrix
				diffusion_ii_audio_adj += diffusion_modal_fusion_ii_matrix
			
			image_ui_embedding, text_ui_embedding, audio_ui_embedding = self.item_item_GCN(R, original_ui_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj)

			sepcial_image_ui_embedding, special_text_ui_embedding, special_audio_ui_embedding, common_embedding = self.gate_attention_fusion(image_ui_embedding, text_ui_embedding, audio_ui_embedding)
			image_prefer_embedding = self.gate_image_modal(content_embedding) 
			text_prefer_embedding = self.gate_text_modal(content_embedding) 
			audio_prefer_embedding = self.gate_audio_modal(content_embedding) 

			sepcial_image_ui_embedding = torch.multiply(image_prefer_embedding, sepcial_image_ui_embedding)
			special_text_ui_embedding = torch.multiply(text_prefer_embedding, special_text_ui_embedding)
			special_audio_ui_embedding = torch.multiply(audio_prefer_embedding, special_audio_ui_embedding)

			side_embedding = (sepcial_image_ui_embedding + special_text_ui_embedding + special_audio_ui_embedding + common_embedding) / 4
			all_embedding = content_embedding + side_embedding
		else:
			if self.modal_fusion == True:
				diffusion_ii_image_adj += diffusion_modal_fusion_ii_matrix
				diffusion_ii_text_adj += diffusion_modal_fusion_ii_matrix


			image_ui_embedding, text_ui_embedding = self.item_item_GCN(R, original_ui_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj=None)
			sepcial_image_ui_embedding, special_text_ui_embedding, common_embedding = self.gate_attention_fusion(image_ui_embedding, text_ui_embedding, audio_ui_embedding=None)
			image_prefer_embedding = self.gate_image_modal(content_embedding) 
			text_prefer_embedding = self.gate_text_modal(content_embedding) 
			sepcial_image_ui_embedding = torch.multiply(image_prefer_embedding, sepcial_image_ui_embedding)
			special_text_ui_embedding = torch.multiply(text_prefer_embedding, special_text_ui_embedding)

			side_embedding = (sepcial_image_ui_embedding + special_text_ui_embedding + common_embedding) / 3
			all_embedding = content_embedding + side_embedding
		
		# split 
		all_embeddings_users, all_embeddings_items = torch.split(all_embedding, [args.user, args.item], dim=0)
		
		return all_embeddings_users, all_embeddings_items, side_embedding, content_embedding


class Model(nn.Module):
	def __init__(self, image_embedding, text_embedding, audio_embedding=None):
		'''
			image_embedding, text_embedding, audio_embedding.shape: torch.Size([6710, 128])
		
		'''
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim))) 
		self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

		self.edgeDropper = SpAdjDropEdge(args.keepRate)

		if args.trans == 1:
			self.image_trans = nn.Linear(args.image_feat_dim, args.latdim)
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		elif args.trans == 0:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))
		else:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		if audio_embedding != None:
			if args.trans == 1:
				self.audio_trans = nn.Linear(args.audio_feat_dim, args.latdim)
			else:
				self.audio_trans = nn.Parameter(init(torch.empty(size=(args.audio_feat_dim, args.latdim))))

		self.image_embedding = image_embedding
		self.text_embedding = text_embedding
		if audio_embedding != None:
			self.audio_embedding = audio_embedding
		else:
			self.audio_embedding = None

		if audio_embedding != None:
			self.modal_weight = nn.Parameter(torch.Tensor([0.3333, 0.3333, 0.3333]))
		else:
			self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
		self.softmax = nn.Softmax(dim=0)

		self.dropout = nn.Dropout(p=0.1)

		self.leakyrelu = nn.LeakyReLU(0.2)
				
	def getItemEmbeds(self):
		return self.iEmbeds
	
	def getUserEmbeds(self):
		return self.uEmbeds
	
	def getImageFeats(self):
		if args.trans == 0 or args.trans == 2:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			return image_feats
		else:
			return self.image_trans(self.image_embedding)
	
	def getTextFeats(self):
		if args.trans == 0:
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
			return text_feats
		else:
			return self.text_trans(self.text_embedding)

	def getAudioFeats(self):
		if self.audio_embedding == None:
			return None
		else:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)
		return audio_feats

	def forward_MM(self, adj, image_adj, text_adj, audio_adj=None):
		'''
			adj: 原始的user-item矩阵, image_adj, text_adj, audio_adj: 生成的user-item矩阵, torch.Size([16018, 16018])
		'''
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		if audio_adj != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)

		weight = self.softmax(self.modal_weight)


		# 对扩散模型生成的图先进行一层图卷积，使生成和原始的embedding进行加权融合
		embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds]) # user embedding , item embedding concat
		embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)

		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		embedsImage = torch.spmm(adj, embedsImage)

		embedsImage_ = torch.concat([embedsImage[:args.user], self.iEmbeds])
		embedsImage_ = torch.spmm(adj, embedsImage_)
		embedsImage += embedsImage_
		
		embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(adj, embedsText)

		embedsText_ = torch.concat([embedsText[:args.user], self.iEmbeds])
		embedsText_ = torch.spmm(adj, embedsText_)
		embedsText += embedsText_

		if audio_adj != None:
			embedsAudioAdj = torch.concat([self.uEmbeds, self.iEmbeds])
			embedsAudioAdj = torch.spmm(audio_adj, embedsAudioAdj)

			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(adj, embedsAudio)

			embedsAudio_ = torch.concat([embedsAudio[:args.user], self.iEmbeds])
			embedsAudio_ = torch.spmm(adj, embedsAudio_)
			embedsAudio += embedsAudio_

		embedsImage += args.ris_adj_lambda * embedsImageAdj # 原始的 + 生成的*权重
		embedsText += args.ris_adj_lambda * embedsTextAdj
		if audio_adj != None:
			embedsAudio += args.ris_adj_lambda * embedsAudioAdj
		if audio_adj == None:
			embedsModal = weight[0] * embedsImage + weight[1] * embedsText
		else:
			embedsModal = weight[0] * embedsImage + weight[1] * embedsText + weight[2] * embedsAudio

		# 融合完成
		embeds = embedsModal
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)

		embeds = embeds + args.ris_lambda * F.normalize(embedsModal)

		return embeds[:args.user], embeds[args.user:]


	def forward_cl_MM(self, adj, image_adj, text_adj, audio_adj=None):
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		if audio_adj != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)

		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		embedsImage = torch.spmm(image_adj, embedsImage)

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(text_adj, embedsText)

		if audio_adj != None:
			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(audio_adj, embedsAudio)

		embeds1 = embedsImage
		embedsLst1 = [embeds1]
		for gcn in self.gcnLayers:
			embeds1 = gcn(adj, embedsLst1[-1])
			embedsLst1.append(embeds1)
		embeds1 = sum(embedsLst1)

		embeds2 = embedsText
		embedsLst2 = [embeds2]
		for gcn in self.gcnLayers:
			embeds2 = gcn(adj, embedsLst2[-1])
			embedsLst2.append(embeds2)
		embeds2 = sum(embedsLst2)

		if audio_adj != None:
			embeds3 = embedsAudio
			embedsLst3 = [embeds3]
			for gcn in self.gcnLayers:
				embeds3 = gcn(adj, embedsLst3[-1])
				embedsLst3.append(embeds3)
			embeds3 = sum(embedsLst3)

		if audio_adj == None:
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:]
		else:
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:], embeds3[:args.user], embeds3[args.user:]

	def reg_loss(self):
		ret = 0
		ret += self.uEmbeds.norm(2).square()
		ret += self.iEmbeds.norm(2).square()
		return ret

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)

class SpAdjDropEdge(nn.Module):
	def __init__(self, keepRate):
		super(SpAdjDropEdge, self).__init__()
		self.keepRate = keepRate

	def forward(self, adj):
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

		newVals = vals[mask] / self.keepRate
		newIdxs = idxs[:, mask]

		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class MultimodalDenoise(nn.Module):
	'''
		多模态降噪模型: 用于预测扩散模型的噪声
		特性： 在扩散的每一步骤中，引入跨模态的引导信息，使得模型能够根据不同模态的特征更好地调整生成的方向。
		例如，利用文本特征引导扩散过程中对物品语义相关属性的生成，利用图像特征引导视觉相关属性的生成，通过设计合适的注意力机制或者额外的引导损失函数等，
		让各模态特征在扩散过程中相互协作、相互制约，以生成更符合多模态综合信息的样本。
	'''	
	def __init__(self, in_dims, out_dims, tiem_emb_size, norm=False, dropout=0.2):
		'''
			初始化
		'''
		super(MultimodalDenoise, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims 
		self.time_emb_dim = tiem_emb_size 
		self.norm = norm 
		self.dropout = nn.Dropout(dropout)
		self.time_embedding_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		self.linear_projrct = nn.Sequential(
			nn.Linear()
		)
	
	def image_modal_project(self, image_feature):
		'''
			图像模态特征提取
		'''	
		pass 
	
	def time_embedding(self, timesteps):
		'''
			Time embedding
		'''
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		#print("time_emb.shape:", time_emb.shape)
		emb = self.emb_layer(time_emb)
		return emb
	
	def forward(self, x_image, x_text, x_audio, timesteps, mess_dropout=True):
		'''
			x_image : 加噪的视觉模态特征 torch.Size([1024, 128])
			x_text : 加噪的视觉模态特征  torch.Size([1024, 768])
			x_audio : 加噪的视觉模态特征 torch.Size([1024, 128])
		'''
		time_emb = self.time_embedding(timesteps)
		if self.norm:
			x_image= F.normalize(x_image)
			x_text = F.normalize(x_text)
			x_audio = F.normalize(x_audio)

		if mess_dropout:
			x_image = self.dropout(x_image)
			x_text = self.dropout(x_text)
			x_audio = self.dropout(x_audio)



class ModalDenoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.2):
		'''
			生成epsilon
			没有Embedding ?
		'''
		super(ModalDenoise, self).__init__()
		self.in_dims = in_dims # 128
		self.out_dims = out_dims # 128
		self.time_emb_dim = emb_size # 10
		# print("self.in_dims:", self.in_dims)
		# print("self.out_dims", self.out_dims)
		# print("self.time_emb_dim:", self.time_emb_dim)
		self.norm = norm
		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		#print("self.in_dims + self.time_emb_dim:", in_dims + self.time_emb_dim) # 4106
		# print("self.in_dims//2:", self.in_dims//2)  # (1024x1034 and 4116x2053) # 2048

		in_features = (in_dims + self.time_emb_dim) 

		self.down_sampling = nn.Sequential(
			nn.Linear(in_features=in_features, out_features=self.in_dims // 2),
			nn.BatchNorm1d(self.in_dims // 2),
			nn.LeakyReLU(),
			nn.Dropout(0.3),
			nn.Linear(in_features=self.in_dims // 2, out_features=self.in_dims//4),
			nn.BatchNorm1d(self.in_dims//4),
			nn.LeakyReLU(),
			nn.Dropout(0.3),
			nn.Linear(in_features=self.in_dims // 4, out_features=self.in_dims//8),
			nn.BatchNorm1d(self.in_dims//8),
			nn.LeakyReLU(),
			nn.Dropout(0.3)

		)

		self.up_sampling = nn.Sequential(
			nn.Linear(in_features=self.in_dims//8, out_features=self.in_dims//4),
			nn.BatchNorm1d(self.in_dims//4),
			nn.LeakyReLU(),
			nn.Dropout(0.3),
			nn.Linear(in_features=self.in_dims//4, out_features=self.in_dims//2),
			nn.BatchNorm1d(self.in_dims // 2),
			nn.LeakyReLU(),
			nn.Dropout(0.3),
			nn.Linear(in_features=self.in_dims//2, out_features=self.in_dims),
			nn.BatchNorm1d(self.in_dims),
			nn.LeakyReLU(),
			nn.Dropout(0.3)
		)


		
		self.drop = nn.Dropout(dropout)
		self.initialize_weights()

	def initialize_weights(self):
			"""
			对down_sampling和up_sampling中的线性层权重和偏差进行初始化
			"""
			for module_seq in [self.down_sampling, self.up_sampling]:
				for layer in module_seq:
					if isinstance(layer, nn.Linear):
						size = layer.weight.size()
						std = np.sqrt(2.0 / (size[0] + size[1]))
						layer.weight.data.normal_(0.0, std)
						layer.bias.data.normal_(0.0, 0.001)

	def forward(self, x, timesteps, mess_dropout=True):
		# print("x.shape:", x.shape) # x.shape: torch.Size([1024, 128])
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		#print("time_emb.shape:", time_emb.shape)
		emb = self.emb_layer(time_emb)
		
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
		#print("x.shape:", x.shape) # tiktok x.shape: torch.Size([1024, 128])   baby: x.shape: torch.Size([1024, 4096])
		# print("emb.shape:", emb.shape)
		h = torch.cat([x, emb], dim=-1)
		#print("h0.shape:", h.shape) # h1.shape: torch.Size([1024, 138])  h0.shape: torch.Size([1024, 4106]
		# dowm sapmling
		h = self.down_sampling(h)
		#print("h2.shape:", h.shape) # h2.shape: torch.Size([1024, 32])
		# up sampling
		h = self.up_sampling(h)

		# x += h 
		# h = torch.cat([x, emb], dim=-1)
		# h = self.down_sampling(h)
		# h = self.up_sampling(h)
		#print("h3.shape:", h.shape) # h3.shape: torch.Size([1024, 128])
		return h


class ModalDenoiseUNet(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		'''
			生成epsilon
			没有Embedding ?
		'''
		super(ModalDenoiseUNet, self).__init__()
		self.in_dims = in_dims # 128
		self.out_dims = out_dims # 128
		self.time_emb_dim = emb_size # 10
		self.norm = norm
		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
		
		in_features = (in_dims + self.time_emb_dim)

		#print("in_features",in_features)
		# 下采样
		self.down_sampling1 = self.mlp(in_features, self.in_dims // 2)
		self.down_pool1 = nn.MaxPool1d(1)

		self.down_sampling2 = self.mlp(self.in_dims // 2, self.in_dims// 4)
		self.down_pool2 = nn.MaxPool1d(1)

		self.down_sampling3 = self.mlp(self.in_dims// 4, self.in_dims// 8)
		self.down_pool3 = nn.MaxPool1d(1)
		
		self.middle_connect = self.mlp(self.in_dims //8, self.in_dims // 8)

		# 上采样
		self.up_sampling1 = self.up_sampling(self.in_dims // 8 + self.in_dims // 8 , self.in_dims // 4)
		self.up_sampling2 = self.up_sampling(self.in_dims // 4 + self.in_dims // 4, self.in_dims // 2)
		self.up_sampling3 = self.up_sampling(self.in_dims // 2  +  self.in_dims // 2, self.in_dims)

		self.drop = nn.Dropout(dropout)
		self.initialize_weights()

	def mlp(self, in_features, out_features):
		#print("in_features, out_features:", in_features, out_features)
		return nn.Sequential(
			nn.Linear(in_features=in_features, out_features=out_features),
			nn.BatchNorm1d(out_features),
			nn.LeakyReLU(),
			nn.Dropout(),
			nn.Linear(in_features=out_features, out_features=out_features),
			nn.BatchNorm1d(out_features),
			nn.LeakyReLU(),
			nn.Dropout() 
		)


	def up_sampling(self, in_features, out_features):
		return nn.Sequential(
			nn.Linear(in_features=in_features, out_features=out_features),
			nn.BatchNorm1d(out_features),
			nn.ReLU(inplace=True)
		)
	

	def initialize_weights(self):
		"""
		对down_sampling、middle_connect和up_sampling中的线性层权重和偏差进行初始化
		"""
		# 遍历下采样模块中的线性层进行初始化
		for down_module in [self.down_sampling1, self.down_sampling2, self.down_sampling3]:
			for layer in down_module:
				if isinstance(layer, nn.Linear):
					size = layer.weight.size()
					std = np.sqrt(2.0 / (size[0] + size[1]))
					layer.weight.data.normal_(0.0, std)
					layer.bias.data.normal_(0.0, 0.001)

		# 对中间连接模块中的线性层进行初始化
		for layer in self.middle_connect:
			if isinstance(layer, nn.Linear):
				size = layer.weight.size()
				std = np.sqrt(2.0 / (size[0] + size[1]))
				layer.weight.data.normal_(0.0, std)
				layer.bias.data.normal_(0.0, 0.001)

		# 遍历上采样模块中的线性层进行初始化
		for up_module in [self.up_sampling1, self.up_sampling2, self.up_sampling3]:
			for layer in up_module:
				if isinstance(layer, nn.Linear):
					size = layer.weight.size()
					std = np.sqrt(2.0 / (size[0] + size[1]))
					layer.weight.data.normal_(0.0, std)
					layer.bias.data.normal_(0.0, 0.001)

		# 对时间嵌入层的线性层进行初始化
		for layer in [self.emb_layer]:
			if isinstance(layer, nn.Linear):
				size = layer.weight.size()
				std = np.sqrt(2.0 / (size[0] + size[1]))
				layer.weight.data.normal_(0.0, std)
				layer.bias.data.normal_(0.0, 0.001)


	def forward(self, x, timesteps, mess_dropout=True):
		# \\print("x.shape:", x.shape) # x.shape: torch.Size([1024, 128])
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		#print("time_emb.shape:", time_emb.shape)
		emb = self.emb_layer(time_emb)
		
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
	
		# print("x.shape:", x.shape) # tiktok x.shape: torch.Size([1024, 128])   baby: x.shape: torch.Size([1024, 4096])
		# print("emb.shape:", emb.shape)
		h = torch.cat([x, emb], dim=-1)
		# Down Sample
		#print("h .shape:", h.shape) #  h .shape: torch.Size([1024, 4106])
		down_x1 = self.down_sampling1(h)
		#print('down_x1.shape:', down_x1.shape) # down_x1.shape: torch.Size([1024, 2048])
		pool_x1 = self.down_pool1(down_x1)
		#print("pool_x1.shape:", pool_x1.shape) # pool_x1.shape: torch.Size([1024, 2048])
		down_x2 = self.down_sampling2(pool_x1)
		#print('down_x2.shape:', down_x2.shape)  # down_x2.shape: torch.Size([1024, 1024])
		pool_x2 = self.down_pool2(down_x2)
		#print("pool_x2.shape:", pool_x2.shape) # pool_x2.shape: torch.Size([1024, 1024])
		down_x3 = self.down_sampling3(pool_x2)
		#print('down_x3.shape:', down_x3.shape)  # down_x3.shape: torch.Size([1024, 512])
		pool_x3 = self.down_pool3(down_x3)
		#print("pool_x3.shape:", pool_x3.shape)  # pool_x3.shape: torch.Size([1024, 512])
		# Middle 
		middle = self.middle_connect(pool_x3)
		#print("middle.shape:", middle.shape) # middle.shape: torch.Size([1024, 512])
		#print("torch.cat([middle, down_x3].shape:", torch.cat([middle, down_x3], dim=1).shape)
        # 上采样过程，融合跳跃连接的特征
		up_x1 = self.up_sampling1(torch.cat([middle, down_x3], dim=1))
		#print("up_x1.shape:", up_x1.shape)
		up_x2 = self.up_sampling2(torch.cat([up_x1, down_x2], dim=1))
		#print("up_x2.shape:", up_x2.shape)
		up_x3 = self.up_sampling3(torch.cat([up_x2, down_x1], dim=1))
		#print("up_x3.shape:", up_x3.shape)
		return up_x3
	



class Denoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		'''
			生成epsilon
			没有Embedding ?
		'''
		super(Denoise, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		self.norm = norm

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

		out_dims_temp = self.out_dims

		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
		
		self.drop = nn.Dropout(dropout)
		self.init_weights()

	def init_weights(self):
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)
		
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)

	def forward(self, x, timesteps, mess_dropout=True):
		#print("x.shape:", x.shape)
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		emb = self.emb_layer(time_emb)
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
		h = torch.cat([x, emb], dim=-1)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)
		#print("h.shape:", h.shape)
		return h


class  LaplaceDiffusion(nn.Module):
	'''
		拉普拉斯扩散：视觉、音频、文本扩散
			
	'''
	pass 



class BernoulliDiffusion(nn.Module):
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		'''
		伯努利：
		缓解用户-物品交互矩阵的稀疏0-1扩散
	
		'''
		super(BernoulliDiffusion, self).__init__()

		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

		self.i = 0
	def get_betas(self):
		'''
			TODO: 在实际应用中，数据中的噪声可能是非线性的，这样的线性噪声生成机制可能会限制模型对真实噪声的拟合能力。
		
		'''
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas) 

	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, model, x_start, steps, sampling_noise=False):
		'''
			这种简单的高斯噪声添加方式可能不适合所有的数据类型和分布。如果数据本身具有非高斯的噪声特性，或者数据的结构对噪声的形式有特殊要求，这样的噪声添加方式可能会导致生成的样本质量下降。
			TODO: 多模态仍采用高斯搞噪声， 而交互图则属于稀疏分布，需要计算以下其数据分布
		
		'''
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)
		
		indices = list(range(self.steps))[::-1]

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
			if sampling_noise:

				# noise = torch.randn_like(x_t)
				noise_prob = torch.rand_like(x_t)
				noise = torch.bernoulli(noise_prob)

				nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
				x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
			else:
				x_t = model_mean
		return x_t


	def q_sample(self, x_start, t, noise=None):
		'''
			标准的前向扩散: 加入噪音
		
		'''
		if noise is None:
			# noise = torch.randn_like(x_start)
			noise_prob = torch.rand_like(x_start)
			noise = torch.bernoulli(noise_prob)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

	def p_mean_variance(self, model, x, t):
		model_output = model(x, t, False)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
		'''
			model: 降噪模型
			x_start:
			 		[tensor([[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					...,
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.]])

			x_start.shape: torch.Size([1024, 6710])

			itmEmbeds: item embedding
			model_feats: 

		'''
		batch_size = x_start.size(0) # 1024
		
		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda() # ts: tensor([2, 3, 3,  ..., 0, 1, 1], device='cuda:0')

		# noise = torch.randn_like(x_start) 
		noise_prob = torch.rand_like(x_start)
		noise = torch.bernoulli(noise_prob)
		# print("noise:", noise)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start
		
		# x_t作为采样后加噪的特征，形状不变
		#print("x_start.shape:", x_start.shape)
		#print("x_t.shape:", x_t.shape) # x_t.shape: torch.Size([1024, 6710]) 
		model_output = model(x_t, ts) #计算模型预测t时刻的噪声: model_output.shape: torch.Size([1024, 6710])

		# mse = self.mean_flat((x_start - model_output) ** 2)
		mse = self.mean_flat((noise - model_output) ** 2)
		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)

		diff_loss = weight * mse

		usr_model_embeds = torch.mm(model_output, model_feats)
		usr_id_embeds = torch.mm(x_start, itmEmbeds)

		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

		return diff_loss, gc_loss
	

	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])


class GaussianDiffusion(nn.Module):
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()
		self.i = 0
		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

	def get_betas(self):
		'''
			TODO: 在实际应用中，数据中的噪声可能是非线性的，这样的线性噪声生成机制可能会限制模型对真实噪声的拟合能力。
		
		'''
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas) 

	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, model, x_start, steps, sampling_noise=False):
		'''
			这种简单的高斯噪声添加方式可能不适合所有的数据类型和分布。如果数据本身具有非高斯的噪声特性，或者数据的结构对噪声的形式有特殊要求，这样的噪声添加方式可能会导致生成的样本质量下降。
			TODO: 多模态仍采用高斯搞噪声， 而交互图则属于稀疏分布，需要计算以下其数据分布
		
		'''
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)
		
		indices = list(range(self.steps))[::-1]

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
			if sampling_noise:

				noise = torch.randn_like(x_t)
				nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
				x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
			else:
				x_t = model_mean
		return x_t


	def q_sample(self, x_start, t, noise=None):
		'''
			标准的前向扩散: 加入噪音 
			Noise(t, s, pos)
			dense diffusion q_sample:x_start: tensor([[3.1504, 1.7197, 4.0469,  ..., 2.2773, 5.6953, 2.9414],
							[2.8691, 3.0664, 1.2949,  ..., 4.4531, 1.7051, 0.3076],
							[3.1055, 2.2812, 5.6992,  ..., 2.7812, 1.9307, 4.4336],
							...,
							[0.8501, 3.2363, 2.4023,  ..., 4.9102, 1.0312, 1.1895],
							[3.0508, 2.5781, 3.0469,  ..., 5.0273, 1.0996, 3.7324],
							[4.1758, 6.7891, 1.5459,  ..., 5.0156, 4.8594, 2.8789]],
						device='cuda:0')
		'''
		#print("dense diffusion q_sample:x_start:", x_start)

		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

	def p_mean_variance(self, model, x, t):
		model_output = model(x, t, False)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
		'''
			model: 降噪模型
			x_start:
			 		[tensor([[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					...,
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.]])

			x_start.shape: torch.Size([1024, 6710])

			itmEmbeds: item embedding
			model_feats: 

		'''
		#print("training_losses:x_start:", x_start)
		batch_size = x_start.size(0) # 1024
		
		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda() # ts: tensor([2, 3, 3,  ..., 0, 1, 1], device='cuda:0')

		noise = torch.randn_like(x_start) 
		# print("noise:", noise)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start
		
		# x_t作为采样后加噪的特征，形状不变
		#print("x_start.shape:", x_start.shape)
		#print("x_t.shape:", x_t.shape) # x_t.shape: torch.Size([1024, 6710]) 
		model_output = model(x_t, ts) #计算模型预测t时刻的噪声: model_output.shape: torch.Size([1024, 6710])

		# mse = self.mean_flat((x_start - model_output) ** 2)
		mse = self.mean_flat((noise - model_output) ** 2)
		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)

		diff_loss = weight * mse
		#
		usr_model_embeds = torch.mm(model_output, model_feats)
		usr_id_embeds = torch.mm(x_start, itmEmbeds)
		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)
		
		# 原始图卷积的模态向量 与 生成的模态向量之间的对比损失，使得生成的模态正样本靠近原始
		model_feat_embedding =  torch.multiply(itmEmbeds, model_feats)
		model_feat_embedding_origin = torch.mm(x_start, model_feat_embedding)
		model_feat_embedding_diffusion = torch.mm(model_output, model_feat_embedding)

		contra_loss = self.infoNCE_loss(model_feat_embedding_origin, model_feat_embedding_diffusion, 0.2)

		return diff_loss, gc_loss, contra_loss
	

	def training_multimodal_feature_diffusion_losses(self, model, x_start):
		'''
			model: 降噪模型

		'''
		self.i += 1
		#print("self.i:", self.i)
		batch_size = x_start.size(0) # 1024
		#print("x_start.shape:", x_start.shape) # x_start.shape: torch.Size([1024, 128])
		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda() # ts: tensor([2, 3, 3,  ..., 0, 1, 1], device='cuda:0')

		noise = torch.randn_like(x_start) 
		# print("noise:", noise)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start
		
		# x_t作为采样后加噪的特征，形状不变
		#print("x_t.shape:", x_t.shape) # x_t.shape: torch.Size([1024, 128]) 
		#print("x_start.shape:", x_start.shape)
		#print("x_t.shape:", x_t.shape) # x_t.shape: torch.Size([1024, 6710]) 
		model_output = model(x_t, ts) #计算模型预测t时刻的噪声: model_output.shape: torch.Size([1024, 6710])
		#print("noise:", noise.shape, "model_output:", model_output.shape) # noise: torch.Size([1024, 128]) model_output: torch.Size([1024, 138])
		mse = self.mean_flat((noise - model_output) ** 2)
		#print("mse:", mse.shape)
		modal_feature_diffusion_loss = mse
		# weight = self.SNR(ts - 1) - self.SNR(ts)
		# weight = torch.where((ts == 0), 1.0, weight)

		# modal_feature_diffusion_loss = weight * mse

		return modal_feature_diffusion_loss
		
			
	def infoNCE_loss(self, view1, view2,  temperature):
		'''
			InfoNCE loss
		'''
		view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
		pos_score = torch.sum((view1 * view2), dim=-1)
		pos_score = torch.exp(pos_score / temperature)

		neg_score = (view1 @ view2.T) / temperature
		neg_score = torch.exp(neg_score).sum(dim=1)
		contrast_loss = -1 * torch.log(pos_score / neg_score).mean()

		return contrast_loss

		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
	



class SparityDiffusion(nn.Module):
	'''
		User-Item Interaction Graph Sparity Diffusion 

	'''
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(SparityDiffusion, self).__init__()

		self.i = 0
		self.alpha_sparity = 0.01
		self.beta_sparity = 0.01
		self.open_noise_adaptive = True
		self.noise_adaptive_factor = 1.0
		self.postive_gain_degree  = 0.5

		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

	def get_betas(self):
		'''
			TODO: 在实际应用中，数据中的噪声可能是非线性的，这样的线性噪声生成机制可能会限制模型对真实噪声的拟合能力。
		
		'''
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas) 

	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, model, x_start, steps, sampling_noise=False):
		'''
			这种简单的高斯噪声添加方式可能不适合所有的数据类型和分布。如果数据本身具有非高斯的噪声特性，或者数据的结构对噪声的形式有特殊要求，这样的噪声添加方式可能会导致生成的样本质量下降。
			TODO: 多模态仍采用高斯搞噪声， 而交互图则属于稀疏分布，需要计算以下其数据分布
		
		'''
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)
		
		indices = list(range(self.steps))[::-1]

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
			if sampling_noise:

				noise = torch.randn_like(x_t)
				nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
				x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
			else:
				x_t = model_mean
		return x_t


	def q_sample(self, x_start, t, noise=None):
		'''
			User-Item稀疏自适应采样
			标准的前向扩散: 加入噪音 
			Noise(t, s, pos)
			We can notice that the user-item interaction matrix is a sparity graph:
			sparity diffusion q_sample:
							x_start: tensor([[0., 0., 0.,  ..., 0., 0., 0.],
										[0., 0., 0.,  ..., 0., 0., 0.],
										[0., 0., 0.,  ..., 0., 0., 0.],
										...,
										[0., 0., 0.,  ..., 0., 0., 0.],
										[0., 0., 0.,  ..., 0., 0., 0.],
										[0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')
		
		'''
		#print("sparity diffusion q_sample:x_start:", x_start)
		if self.open_noise_adaptive:
			'''
				计算自适应噪声系数:
					1. 计算对当前的batch级别的稀疏度 noise_adaptive_penalty_factor:1 - 交互图求和 / (batch_size x items number) 目的是用于宏观的控制batch与batch之间的稀疏噪声比例,
					 	 noise_adaptive_penalty_factor 自适应的噪声惩罚因子越大说明当前batch的交互越稀疏,那么batch级别的整体噪声对比其他batch会更大
					2. 超参数:alpha_sparity用于控制噪声缩放的比例, beta_sparity用于控制噪声衰减的程度
					3. 计算每个batch内部原始的交互正样本mask: 通过降低batch内原始交互的的噪声,保留原始交互的信息


			'''
			batch_size = x_start.shape[0]
			item_size = x_start.shape[1]
			#计算对当前的batch级别的稀疏度
			#print("t:", t)
			#print("t.shape:", t.shape)
			#print("x_start.sum():", x_start.sum())
			batch_noise_adaptive_penalty_factor  = 1 - (x_start.sum() / (batch_size * item_size))
			#print("batch_noise_adaptive_penalty_factor:", batch_noise_adaptive_penalty_factor)
			noise_coe = self.alpha_sparity * (1 + batch_noise_adaptive_penalty_factor) * torch.exp(-1.0 * self.beta_sparity * t) # batch
			# 计算每个batch内部原始的交互正样本mask:
			ones_tensor = torch.ones_like(x_start)
			batch_postive_position_mask_matirx = torch.where(x_start == 0, ones_tensor - x_start, self.postive_gain_degree * x_start)
			#print("noise_coe:", noise_coe)
			#print("batch_postive_position_mask_matirx:", batch_postive_position_mask_matirx)
			noise_coe = noise_coe.unsqueeze(1)
			# print("noise_coe.unsqueeze:", noise_coe)
			noise_coe =  noise_coe * batch_postive_position_mask_matirx 
			

		if noise is None:
			noise = torch.randn_like(x_start)
			# print("dense noise:", noise)
		# print("dense noise:", noise)
		# print("sparity noise:", noise_coe)
		noise = noise * noise_coe
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise 

	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

	def p_mean_variance(self, model, x, t):
		model_output = model(x, t, False)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
		'''
			model: 降噪模型
			x_start:
			 		[tensor([[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					...,
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.]])

			x_start.shape: torch.Size([1024, 6710])

			itmEmbeds: item embedding
			model_feats: 

		'''
		#print("training_losses:x_start:", x_start)
		batch_size = x_start.size(0) # 1024
		
		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda() # ts: tensor([2, 3, 3,  ..., 0, 1, 1], device='cuda:0')

		noise = torch.randn_like(x_start) 
		# print("noise:", noise)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start
		
		# x_t作为采样后加噪的特征，形状不变
		#print("x_start.shape:", x_start.shape)
		#print("x_t.shape:", x_t.shape) # x_t.shape: torch.Size([1024, 6710]) 
		model_output = model(x_t, ts) #计算模型预测t时刻的噪声: model_output.shape: torch.Size([1024, 6710])

		# mse = self.mean_flat((x_start - model_output) ** 2)
		mse = self.mean_flat((noise - model_output) ** 2)
		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)

		diff_loss = weight * mse
		#
		usr_model_embeds = torch.mm(model_output, model_feats)
		usr_id_embeds = torch.mm(x_start, itmEmbeds)
		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)
		
		# 原始图卷积的模态向量 与 生成的模态向量之间的对比损失，使得生成的模态正样本靠近原始
		model_feat_embedding =  torch.multiply(itmEmbeds, model_feats)
		model_feat_embedding_origin = torch.mm(x_start, model_feat_embedding)
		model_feat_embedding_diffusion = torch.mm(model_output, model_feat_embedding)

		contra_loss = self.infoNCE_loss(model_feat_embedding_origin, model_feat_embedding_diffusion, 0.2)

		return diff_loss, gc_loss, contra_loss
	

			
	def infoNCE_loss(self, view1, view2,  temperature):
		'''
			InfoNCE loss
		'''
		view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
		pos_score = torch.sum((view1 * view2), dim=-1)
		pos_score = torch.exp(pos_score / temperature)

		neg_score = (view1 @ view2.T) / temperature
		neg_score = torch.exp(neg_score).sum(dim=1)
		contrast_loss = -1 * torch.log(pos_score / neg_score).mean()

		return contrast_loss

		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
	