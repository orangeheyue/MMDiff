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
	def __init__(self, image_embedding, text_embedding, audio_embedding=None):
		super(GCNModel, self).__init__()
		
		self.sparse = True
		self.gcn_layer_num = 2
		self.edgeDropper = SpAdjDropEdge(args.keepRate)
		self.reg_weight =  1e-04
		self.batch_size = 1024


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
			self.image_modal_project = nn.Sequential(
				nn.Linear(in_features=self.image_embedding.shape[1], out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU(),
				nn.Dropout()
			)

		if self.text_embedding is not None:
			self.text_modal_project = nn.Sequential(
				nn.Linear(in_features=self.text_embedding.shape[1], out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU(),
				nn.Dropout()
			)

		if self.audio_embedding is not None:
			self.audio_modal_project = nn.Sequential(
				nn.Linear(in_features=self.audio_embedding.shape[1], out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU(),
				nn.Dropout()
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



	def init_modal_weight(self):
		'''
			初始化模型权重
		'''	

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
			image_modal_feature = self.image_modal_project(self.image_embedding)
		return image_modal_feature

	def getTextFeats(self):
		'''
			获取文本模态特征
		'''
		if self.text_embedding is not None:
			text_modal_feature = self.text_modal_project(self.text_embedding)
		return text_modal_feature
	
	def getAudioFeats(self):
		'''
			获取音频模态特征
		'''
		if self.audio_embedding is not None:
			audio_modal_feature = self.audio_modal_project(self.audio_embedding)
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
		'''
		#print("original_ui_adj:", original_ui_adj, "diffusion_ui_adj:", diffusion_ui_adj)
		adj = original_ui_adj + diffusion_ui_adj # 
		cat_embedding = torch.cat([self.item_id_embedding.weight, self.user_embedding.weight], dim=0)

		all_embeddings = [cat_embedding]
		for i in range(self.gcn_layer_num):
			temp_embeddings = torch.sparse.mm(adj, cat_embedding)
			cat_embedding = temp_embeddings
			all_embeddings += [cat_embedding]
		
		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
		content_embedding = all_embeddings

		return content_embedding

	def item_item_GCN(self, original_ui_adj, R, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj=None):
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
		pos_scores = torch.sum(torch.mul(anc_embeds, pos_embeds), dim=1)  # 计算用户与正样本物品的点积之和，作为正样本得分，形状为 [batch_size]
		neg_scores = torch.sum(torch.mul(anc_embeds, neg_embeds), dim=1)  # 计算用户与负样本物品的点积之和，作为负样本得分，形状为 [batch_size]

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
	
	
	def infoNCE_loss(self, view1, view2, nodes, temperature):
		'''
			InfoNCE loss
		'''
		view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
		view1, view2 = view1[nodes], view2[nodes]
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


	def forward(self,R, original_ui_adj, diffusion_ui_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj=None):
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
		'''
		# 多模态特征提取与fusion
	
		content_embedding = self.user_item_GCN(original_ui_adj, diffusion_ui_adj)
		#print("user-item gcn-------->content_embedding.shape", content_embedding.shape) # torch.Size([16018, 64])

		if args.data == 'tiktok':
			image_ui_embedding, text_ui_embedding, audio_ui_embedding = self.item_item_GCN(original_ui_adj, R, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj)

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
			image_ui_embedding, text_ui_embedding = self.item_item_GCN(original_ui_adj, R, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj=None)
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


class ModalDenoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
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

		self.down_sampling = nn.Sequential(
			nn.Linear(in_features=(self.in_dims + self.time_emb_dim), out_features=self.in_dims//2),
			nn.BatchNorm1d(self.in_dims//2),
			nn.LeakyReLU(),
			nn.Dropout(),
			nn.Linear(in_features=self.in_dims//2, out_features=self.in_dims//4),
			nn.BatchNorm1d(self.in_dims//4),
			nn.LeakyReLU(),
			nn.Dropout()
		)

		self.up_sampling = nn.Sequential(
			nn.Linear(in_features=self.in_dims//4, out_features=self.in_dims//2),
			nn.BatchNorm1d(self.in_dims//2),
			nn.LeakyReLU(),
			nn.Dropout(),
			nn.Linear(in_features=self.in_dims//2, out_features=self.in_dims),
			nn.BatchNorm1d(self.in_dims),
			nn.LeakyReLU(),
			nn.Dropout()
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
		# print("x.shape:", x.shape) # x.shape: torch.Size([1024, 128])
		# print("emb.shape:", emb.shape)
		h = torch.cat([x, emb], dim=-1)
		#print("h0.shape:", h.shape) # h1.shape: torch.Size([1024, 138])
		# dowm sapmling
		h = self.down_sampling(h)
		#print("h2.shape:", h.shape) # h2.shape: torch.Size([1024, 32])
		# up sampling
		h = self.up_sampling(h)
		#print("h3.shape:", h.shape) # h3.shape: torch.Size([1024, 128])
		return h


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


class GaussianDiffusion(nn.Module):
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

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
		
		'''
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

		mse = self.mean_flat((x_start - model_output) ** 2)

		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)

		diff_loss = weight * mse

		usr_model_embeds = torch.mm(model_output, model_feats)
		usr_id_embeds = torch.mm(x_start, itmEmbeds)

		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

		return diff_loss, gc_loss
	

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
		


		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
	



	