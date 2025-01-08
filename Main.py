import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, GaussianDiffusion, BernoulliDiffusion, Denoise, ModalDenoise, GCNModel
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
import setproctitle
from scipy.sparse import coo_matrix


class Coach:
	def __init__(self, handler):
		self.handler = handler
		self.knn_k = 5 # TODO?
		self.sparse = True
		self.modal_fusion = False

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')

		recallMax = 0
		ndcgMax = 0
		precisionMax = 0
		bestEpoch = 0

		log('Model Initialized')

		for ep in range(0, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses = self.testEpoch()
				if (reses['Recall'] > recallMax):
					recallMax = reses['Recall']
					ndcgMax = reses['NDCG']
					precisionMax = reses['Precision']
					bestEpoch = ep
				log(self.makePrint('Test', ep, reses, tstFlag))
			print()
		print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax, ' , Precision', precisionMax)

	def prepareModel(self):
		# 主模型：进行图卷积和对比学习的模型
		if args.data == 'tiktok':
			self.image_embedding, self.text_embedding, self.audio_embedding = self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach()
			self.model = GCNModel(self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach(), modal_fusion=self.modal_fusion).cuda()
		else:
			self.image_embedding, self.text_embedding = self.handler.image_feats.detach(), self.handler.text_feats.detach()
			self.model = GCNModel(self.handler.image_feats.detach(), self.handler.text_feats.detach(), modal_fusion=self.modal_fusion).cuda()
		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

		# 扩散模型：辅助模型，用来进行生成特征和交互图，计算扩散损失 
		# TODO: 这里还是原本的DDPM，用于生成密集的密集型特征，但是对于交互矩阵等稀疏型特征，我们可以设计一种新的稀疏型Diffusion: Sparse Diffusion 
		self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda() 
		# self.diffusion_model = BernoulliDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()
		
		# 扩散模型中的降噪模型，用于预测反向生成的噪音
		# out_dims = eval(args.dims) + [args.item]
		# in_dims = out_dims[::-1]
		out_dims = self.image_embedding.shape[0]
		in_dims = self.image_embedding.shape[0]
		#print("in_dims:",in_dims, "out_dims:", out_dims)
		self.denoise_model_image = ModalDenoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_image = torch.optim.Adam(self.denoise_model_image.parameters(), lr=args.lr, weight_decay=0)

		# out_dims = eval(args.dims) + [args.item]
		# in_dims = out_dims[::-1]
		out_dims = self.image_embedding.shape[0]
		in_dims = self.image_embedding.shape[0]
		self.denoise_model_text = ModalDenoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_text = torch.optim.Adam(self.denoise_model_text.parameters(), lr=args.lr, weight_decay=0)

		if args.data == 'tiktok':

			# out_dims = eval(args.dims) + [args.item]
			# in_dims = out_dims[::-1]
			out_dims = self.image_embedding.shape[0]
			in_dims = self.image_embedding.shape[0]
			self.denoise_model_audio = ModalDenoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
			self.denoise_opt_audio = torch.optim.Adam(self.denoise_model_audio.parameters(), lr=args.lr, weight_decay=0)
		
		'''
			实例化多模态特征扩散降噪模型
		'''
		in_dims = args.image_feats_dim
		out_dims= args.image_feats_dim
		# print("in_dims:", in_dims, "out_dims:", out_dims) # in_dims: [6710] out_dims: [6710]
		self.image_modal_denoise_model = ModalDenoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.image_modal_denoise_optimizer = torch.optim.Adam(self.image_modal_denoise_model.parameters(), lr=args.lr, weight_decay=0)

		in_dims = args.text_feats_dim
		out_dims= args.text_feats_dim
		self.text_modal_denoise_model = ModalDenoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.text_modal_denoise_optimizer = torch.optim.Adam(self.text_modal_denoise_model.parameters(), lr=args.lr, weight_decay=0)

		if args.data == 'tiktok':
					in_dims = args.audio_feats_dim
					out_dims= args.audio_feats_dim
					self.audio_modal_denoise_model = ModalDenoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
					self.audio_modal_denoise_optimizer = torch.optim.Adam(self.audio_modal_denoise_model.parameters(), lr=args.lr, weight_decay=0)


	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def buildUIMatrix(self, u_list, i_list, edge_list):
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)

		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)

		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()
	
	def buildItem2ItemMatrix(self, feature):
		'''
			根据模态的特征计算Item-Item矩阵
		'''
		feature_embedding = torch.nn.Embedding.from_pretrained(feature, freeze=False)
		feature_embedding = feature_embedding.weight.detach()
		feature_norm = feature.div(torch.norm(feature_embedding, p=2, dim=-1, keepdim=True))
		sim_adj = torch.mm(feature_norm, feature_norm.transpose(1, 0))
		sim_adj_sparse = build_knn_normalized_graph(sim_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
		
		return sim_adj, sim_adj_sparse
		

	def trainEpoch(self):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		epLoss, epRecLoss, epClLoss = 0, 0, 0
		epDiLoss = 0
		epDiLoss_image, epDiLoss_text = 0, 0
		if args.data == 'tiktok':
			epDiLoss_audio = 0

		# record multimodal feature diffusion loss for one epoch 
		epDiLoss_image_modal, epDiLoss_text_modal = 0, 0 
		if args.data == 'tiktok':
			epDiLoss_audio_modal = 0 

		steps = trnLoader.dataset.__len__() // args.batch

		diffusionLoader = self.handler.diffusionLoader
		multimodalFeatureLoader = self.handler.multimodalFeatureLoader

		'''
			多模态特征Diffusion
		'''

		for i, data in enumerate(multimodalFeatureLoader):
			'''
				
			'''
			if len(data)  > 2 :
				image_batch, text_batch, audio_batch  = data  
				image_batch, text_batch, audio_batch = image_batch.cuda(), text_batch.cuda(), audio_batch.cuda()
				
				
			else:
				image_batch, text_batch  = data 
				image_batch, text_batch = image_batch.cuda(), text_batch.cuda()
			# print("image_batch.shape:", image_batch.shape, "text_batch.shape:", text_batch.shape) #Baby:image_batch.shape: torch.Size([1024, 4096]) text_batch.shape: torch.Size([1024, 1024])
			# print("image_batch.shape:", image_batch.shape, "text_batch.shape:", text_batch.shape, "audio_batch.shape:", audio_batch.shape)
			self.image_modal_denoise_optimizer.zero_grad()
			self.text_modal_denoise_optimizer.zero_grad()
			if args.data == 'tiktok':
				self.audio_modal_denoise_optimizer.zero_grad()
			# caculate the diffusion mse loss 
			# TODO: how to deal with multimodal ? parallel or cross?
			#print("image_batch.shape:", image_batch.shape) # image_batch.shape: torch.Size([1024, 128])
			image_modal_difussion_loss = self.diffusion_model.training_multimodal_feature_diffusion_losses(
				model=self.image_modal_denoise_model,
				x_start=image_batch
			) 
			#print("image diffusion done------->")
			#print("text_batch.shape:", text_batch.shape)
			text_modal_diffusion_loss = self.diffusion_model.training_multimodal_feature_diffusion_losses(
				model=self.text_modal_denoise_model,
				x_start=text_batch 
			)
			#print("audio_batch.shape:", audio_batch.shape)
			if args.data == 'tiktok':
				audio_modal_diffusion_loss = self.diffusion_model.training_multimodal_feature_diffusion_losses(
				model=self.audio_modal_denoise_model,
				x_start=audio_batch
			)

			image_modal_difussion_loss = image_modal_difussion_loss.mean()
			text_modal_diffusion_loss = text_modal_diffusion_loss.mean()


			epDiLoss_image_modal += image_modal_difussion_loss.item() 
			epDiLoss_text_modal += text_modal_diffusion_loss.item() 


			if args.data == 'tiktok':
				audio_modal_diffusion_loss = audio_modal_diffusion_loss.mean()
				epDiLoss_audio_modal += audio_modal_diffusion_loss.item()

				loss = image_modal_difussion_loss + text_modal_diffusion_loss + audio_modal_diffusion_loss
			else:
				loss = image_modal_difussion_loss + text_modal_diffusion_loss

			loss.backward()

			self.image_modal_denoise_optimizer.step()
			self.text_modal_denoise_optimizer.step()
			if args.data == 'tiktok':
				self.audio_modal_denoise_optimizer.step()

			log('Multimodal Feature Diffusion Step %d/%d' % (i, multimodalFeatureLoader.dataset.__len__() // args.batch), save=False, oneline=True)
		
		log('Multimodal Feature Diffusion Finish in one epoch' + '\n')
		log('Start to generate Multimodal Diffusion Feature Representation' + '\n')


		with torch.no_grad():
			'''
				反向推理生成多模态统一表征, 和 Item-Item Graph Inference
					image_batch.shape: torch.Size([1024, 128]) denoised_image_batch.shape: torch.Size([1024, 128])
					text_batch.shape: torch.Size([1024, 768]) denoised_text_batch.shape: torch.Size([1024, 768])
					audio_batch.shape: torch.Size([1024, 128]) denoised_audio_batch.shape: torch.Size([1024, 128])

					self.image_modal_diffusion_representation.shape: torch.Size([6710, 128])
					self.text_modal_diffusion_representation.shape: torch.Size([6710, 768])

					self.image_II_matrix: tensor(indices=tensor([[   0,    0,    0,  ..., 6709, 6709, 6709],
														[   0, 5877, 1526,  ...,  327, 1313,  338]]),
										values=tensor([0.1697, 0.1057, 0.0953,  ..., 0.0883, 0.0927, 0.0872]),
										device='cuda:0', size=(6710, 6710), nnz=67100, layout=torch.sparse_coo)
			'''

			image_modal_diffusion_representation_list = []
			text_modal_diffusion_representation_list = []
			if args.data == 'tiktok':
				audio_modal_diffusion_representation_list = []

			for _, data in enumerate(multimodalFeatureLoader):
				if len(data) > 2:
					image_batch, text_batch, audio_batch  = data  
					image_batch, text_batch, audio_batch = image_batch.cuda(), text_batch.cuda(), audio_batch.cuda()
				else:
					image_batch, text_batch  = data 
					image_batch, text_batch = image_batch.cuda(), text_batch.cuda()
				#print("text_batch:.shape", text_batch.shape)
				# 生成的图像batch
				denoised_image_batch = self.diffusion_model.p_sample(self.image_modal_denoise_model, image_batch, args.sampling_steps, args.sampling_noise)
				# print("image_batch.shape:", image_batch.shape, "denoised_image_batch.shape:", denoised_image_batch.shape)
				# 生成的文本batch
				denoised_text_batch = self.diffusion_model.p_sample(self.text_modal_denoise_model, text_batch, args.sampling_steps, args.sampling_noise)		
				#print("denoised_text_batch.shape:", denoised_text_batch.shape)
				# print("text_batch.shape:", text_batch.shape, "denoised_text_batch.shape:", denoised_text_batch.shape)		
				# 生成的音频batch
				if args.data == 'tiktok':
					denoised_audio_batch = self.diffusion_model.p_sample(self.audio_modal_denoise_model, audio_batch, args.sampling_steps, args.sampling_noise)
					# print("audio_batch.shape:", image_batch.shape, "denoised_audio_batch.shape:", denoised_image_batch.shape)

				image_modal_diffusion_representation_list.append(denoised_image_batch)
				text_modal_diffusion_representation_list.append(denoised_text_batch)
				if args.data == 'tiktok':
					audio_modal_diffusion_representation_list.append(denoised_audio_batch)

			# 生成Item模态特征表征 TODO: 还可以做点其他的变换
			self.image_modal_diffusion_representation = torch.concat(image_modal_diffusion_representation_list) #  torch.Size([6710, 128])
			self.text_modal_diffusion_representation = torch.concat(text_modal_diffusion_representation_list)   # torch.Size([6710, 768])

			#print("self.image_embedding:", self.image_embedding)
			# print("image_modal_diffusion_representation:", self.image_modal_diffusion_representation)
			# print("nn.Embedding.from_pretrained(self.v_feat, freeze=False):", torch.nn.Embedding.from_pretrained(self.image_embedding, freeze=False).weight.detach())
			# print("self.image_modal_diffusion_representation.shape:", self.image_modal_diffusion_representation.shape)
			# print("self.text_modal_diffusion_representation.shape:", self.text_modal_diffusion_representation.shape)
			# 生成Item2Item Graph
			# diffusion Item-Item Graph
			self.image_modal_diffusion_representation += self.image_embedding
			self.text_modal_diffusion_representation += self.text_embedding 
			

			self.image_II_matrix_dense, self.image_II_matrix = self.buildItem2ItemMatrix(self.image_modal_diffusion_representation)
			self.text_II_matrix_dense, self.text_II_matrix = self.buildItem2ItemMatrix(self.text_modal_diffusion_representation)

		
			# print("self.image_II_matrix:", self.image_II_matrix)
			self.modal_fusion_II_matrix = self.image_II_matrix + self.text_II_matrix
			if args.data == 'tiktok':
							self.audio_modal_diffusion_representation = torch.concat(audio_modal_diffusion_representation_list)
							self.audio_modal_diffusion_representation += self.audio_embedding
							self.audio_II_matrix_dense, self.audio_II_matrix = self.buildItem2ItemMatrix(self.audio_modal_diffusion_representation)
							
							self.modal_fusion_II_matrix = self.image_II_matrix + self.text_II_matrix + self.audio_II_matrix
							self.modal_fusion_II_matrix_dense = self.image_II_matrix_dense + self.text_II_matrix_dense + self.audio_II_matrix_dense

			# origin Item-Item Graph 
			self.image_II_origin_matrix_dense, self.image_II_origin_materix = self.buildItem2ItemMatrix(self.image_embedding) 
			self.text_II_origin_matrix_dense, self.text_II_origin_materix = self.buildItem2ItemMatrix(self.text_embedding)	
			self.audio_II_origin_matrix_dense, self.audio_II_origin_materix = self.buildItem2ItemMatrix(self.audio_embedding)	

			self.image_II_matrix += self.image_II_origin_materix
			self.text_II_matrix += self.text_II_origin_materix
			self.audio_II_matrix += self.audio_II_origin_materix 
			# self.image_II_origin_matrix_dense += self.image_II_matrix 
			
			# self.text_II_origin_matrix_dense += self.text_II_matrix 
			
			# self.audio_II_origin_matrix_dense += self.audio_II_matrix 
			'''
				self.image_II_matrix.shape: torch.Size([6710, 6710])
				self.text_II_matrix.shape: torch.Size([6710, 6710])
				self.audio_II_matrix.shape: torch.Size([6710, 6710])
			'''
			# print("self.image_II_matrix.shape:", self.image_II_matrix.shape)
			# print("self.text_II_matrix.shape:", self.text_II_matrix.shape)
			# print("self.audio_II_matrix.shape:",self.audio_II_matrix.shape)
			
			log('Generate Multimodal Diffusion Feature Representation Done!')

			log('Generate Multimodal Item Item Graph Done!')



		# 交互图的扩散与生成
		for i, batch in enumerate(diffusionLoader):
			'''
			batch: [tensor([[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					...,
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.]]), tensor([5262, 5684, 5341,  ..., 2287, 5615, 1071])]
				batch_item.shape: torch.Size([1024, 6710])
				batch_index.shape: torch.Size([1024])
			'''
			#print("i:", i,'\n')
			# print("batch:", batch)
			batch_item, batch_index = batch
			# print("batch_item.shape:", batch_item.shape)
			# print("batch_index.shape:", batch_index.shape)
			batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

			iEmbeds = self.model.getItemEmbeds().detach()
			uEmbeds = self.model.getUserEmbeds().detach()

			image_feats = self.model.getImageFeats().detach()
			text_feats = self.model.getTextFeats().detach()
			if args.data == 'tiktok':
				audio_feats = self.model.getAudioFeats().detach()

			self.denoise_opt_image.zero_grad()
			self.denoise_opt_text.zero_grad()
			if args.data == 'tiktok':
				self.denoise_opt_audio.zero_grad()
			'''
				image_feats.shape: torch.Size([6710, 64]), text_feats.shape: torch.Size([6710, 64]), audio_feats.shape: torch.Size([6710, 64])
			'''
			# print("image_feats:", image_feats)
			# print("image_feats.shape:", image_feats.shape)
			# print("text_feats:", image_feats)
			# print("text_feats.shape:", image_feats.shape)
			# print("audio_feats:", image_feats)
			# print("audio_feats.shape:", image_feats.shape)

			diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats)
			
			diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(self.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats)
			if args.data == 'tiktok':
				diff_loss_audio, gc_loss_audio = self.diffusion_model.training_losses(self.denoise_model_audio, batch_item, iEmbeds, batch_index, audio_feats)

			loss_image = diff_loss_image.mean() + gc_loss_image.mean() * args.e_loss
			loss_text = diff_loss_text.mean() + gc_loss_text.mean() * args.e_loss
			if args.data == 'tiktok':
				loss_audio = diff_loss_audio.mean() + gc_loss_audio.mean() * args.e_loss


			epDiLoss_image += loss_image.item()
			epDiLoss_text += loss_text.item()
			if args.data == 'tiktok':
				epDiLoss_audio += loss_audio.item()

			if args.data == 'tiktok':
				loss = loss_image + loss_text + loss_audio
			else:
				loss = loss_image + loss_text
			loss = loss_image
			loss.backward()

			self.denoise_opt_image.step()
			self.denoise_opt_text.step()
			if args.data == 'tiktok':
				self.denoise_opt_audio.step()

			log('Diffusion Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch), save=False, oneline=True)

		log('')
		log('Start to re-build UI matrix')

		with torch.no_grad():

			u_list_image = []
			i_list_image = []
			edge_list_image = []

			u_list_text = []
			i_list_text = []
			edge_list_text = []

			if args.data == 'tiktok':
				u_list_audio = []
				i_list_audio = []
				edge_list_audio = []

			for _, batch in enumerate(diffusionLoader):
				batch_item, batch_index = batch
				batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

				# image 
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model_image, batch_item, args.sampling_steps, args.sampling_noise)
				# print("denoised_batch.shape", denoised_batch.shape) # denoised_batch.shape torch.Size([1024, 6710])
				'''
					denoised_batch tensor([[-0.0065,  0.0102,  0.0007,  ..., -0.0208,  0.0343, -0.0176],
						[-0.0034,  0.0129, -0.0017,  ..., -0.0308,  0.0387, -0.0202],
						[-0.0034,  0.0091,  0.0051,  ..., -0.0229,  0.0348, -0.0109],
						...,
						[ 0.0031,  0.0098,  0.0059,  ..., -0.0164,  0.0391, -0.0168],
						[-0.0032,  0.0146, -0.0023,  ..., -0.0251,  0.0329, -0.0150],
						[ 0.0014,  0.0059, -0.0020,  ..., -0.0235,  0.0317, -0.0109]],
					device='cuda:0')

					top_item: 
					tensor([[0.0619, 0.0584, 0.0567,  ..., 0.0518, 0.0499, 0.0493],
					[0.0667, 0.0618, 0.0604,  ..., 0.0505, 0.0503, 0.0493],
					[0.0668, 0.0637, 0.0608,  ..., 0.0502, 0.0502, 0.0500],
					...,
					[0.0653, 0.0624, 0.0621,  ..., 0.0499, 0.0498, 0.0493],
					[0.0666, 0.0651, 0.0591,  ..., 0.0543, 0.0512, 0.0511],
					[0.0644, 0.0603, 0.0576,  ..., 0.0527, 0.0518, 0.0517]],
					device='cuda:0')
				 
				   indices_: 
				   tensor([[4679, 5724,  964,  ...,  535, 2544, 1084],
					[5724, 4679, 2106,  ...,  964, 5668,  535],
					[5724, 4679, 6520,  ..., 4812, 2995,  964],
					...,
					[4679, 6520, 5724,  ..., 3543, 2564,  964],
					[5724, 6520, 4679,  ..., 2995, 2544, 5340],
					[5724, 6520, 4679,  ..., 1899, 4812, 2106]], device='cuda:0')


					self.image_II_matrix.shape: torch.Size([6710, 6710])
					self.text_II_matrix.shape: torch.Size([6710, 6710])
					self.audio_II_matrix.shape: torch.Size([6710, 6710])

				mm_value: TODO: 左上三角矩阵
                tensor([[1.0000, 0.6342, 0.5354, 0.4215, 0.4063, 0.3820, 0.3693],
                        [1.0000, 0.4390, 0.4328, 0.3986, 0.3962, 0.3823, 0.3818],
                        [1.0000, 0.4058, 0.3858, 0.3679, 0.3567, 0.3337, 0.3334],
                        [1.0000, 0.4111, 0.3943, 0.3858, 0.3618, 0.3527, 0.3473],
                        [1.0000, 0.3759, 0.3712, 0.3680, 0.3475, 0.3469, 0.3200]],
                    device='cuda:0')
                mm_indices: 
                tensor([[ 102,   93,  553, 3404, 6761, 1992,  426],
                        [   9, 2662, 6624, 1217, 4855, 1609,  166],
                        [ 171,  353, 2161,  304, 4680, 6501,  495],
                        [ 187, 4544,  572, 3441, 6797, 2323, 5547],
                        [  29, 2801, 3917, 5032, 1382, 5644, 2307]], device='cuda:0')

                mm_value.flatten(): 
                tensor([1.0000, 0.6342, 0.5354, 0.4215, 0.4063, 0.3820, 0.3693, 1.0000, 0.4390,
                        0.4328, 0.3986, 0.3962, 0.3823, 0.3818, 1.0000, 0.4058, 0.3858, 0.3679,
                        0.3567, 0.3337, 0.3334, 1.0000, 0.4111, 0.3943, 0.3858, 0.3618, 0.3527,
                        0.3473, 1.0000, 0.3759, 0.3712, 0.3680, 0.3475, 0.3469, 0.3200],
                    device='cuda:0')
				'''
				# print("args.rebuild_k:", args.rebuild_k)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)
				#print("top_item:", top_item, "indices_:", indices_)
				#print("batch_index.shape[0]:", batch_index.shape[0], "indices_[i].shape[0]:", indices_[i].shape[0]) # batch_index.shape[0]: 1024 indices_[i].shape[0]: 10
				
		
				# # hyper parameters：
				# self.latent_interest_topk = args.rebuild_k
				# self.high_order_latent_interest_topk = self.latent_interest_topk + 2
				for i in range(batch_index.shape[0]):
				# 	# 每一人对每一个物品
				# 	# a user id 
				# 	latent_intertest_items = indices_[i] # 潜在感兴趣的物品id
				# 	#print("int(batch_index[i].cpu().numpy():", int(batch_index[i].cpu().numpy()), "int(indices_[i][j].cpu().numpy()):", indices_[i][j].cpu().numpy())
				# 	# 计算当前这个用户10个潜在感兴趣的物品中的相似度
				# 	#print("latent_intertest_items:", latent_intertest_items) # tensor([4679, 5724,  964, 6520, 5668, 2106, 3135,  535, 2544, 1084], device='cuda:0')
				# 	#print("self.image_II_matrix_dense.shape:", self.image_II_matrix_dense.shape)
				# 	#print("self.image_II_matrix[latent_intertest_items]:", self.image_modal_diffusion_representation[latent_intertest_items])
				# 	latent_multimodal_items_sim =  torch.multiply(self.image_II_matrix_dense[latent_intertest_items], self.text_II_matrix_dense[latent_intertest_items]) # (10, 6710) * (10, 6710) = (10, 6710)
				# 	if args.data == 'tiktok': 
				# 		latent_multimodal_items_sim = torch.multiply(latent_multimodal_items_sim, self.audio_II_matrix_dense[latent_intertest_items]) # (10, 6710) * (10, 6710) = (10, 6710)
				# 	# 根据当前这个用户10个潜在感兴趣的物品中的相似度找到Top的物品：(历史真实交互的物品的扩散降噪生成后的平均概率作为阈值， 只要不小于该阈值，则认为扩散生成的其他物品为当前用户潜在感兴趣的物品)
				# 	latent_multimodal_items_prob, latent_multimodal_items_index = torch.topk(latent_multimodal_items_sim, self.latent_interest_topk, dim=-1) # (10, 10)
				# 	high_order_items_prob, high_order_items_index = torch.topk(latent_multimodal_items_prob.flatten(), self.high_order_latent_interest_topk)
				# 	# TODO：筛选策略
				# 	high_order_latent_interest_items = latent_multimodal_items_index.flatten()[high_order_items_index]
				# 	for item in high_order_latent_interest_items:
				# 		u_list_image.append(int(batch_index[i].cpu().numpy())) # uid 3226
				# 		i_list_image.append(int(item.item()))
				# 		edge_list_image.append(1.0) 


					for j in range(indices_[i].shape[0]): 
						u_list_image.append(int(batch_index[i].cpu().numpy())) # uid 3226
						i_list_image.append(int(indices_[i][j].cpu().numpy())) # item id 5724
						edge_list_image.append(1.0) 

				# text
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model_text, batch_item, args.sampling_steps, args.sampling_noise)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)			
				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 

						u_list_text.append(int(batch_index[i].cpu().numpy()))
						i_list_text.append(int(indices_[i][j].cpu().numpy()))
						edge_list_text.append(1.0)

				if args.data == 'tiktok':
					# audio
					denoised_batch = self.diffusion_model.p_sample(self.denoise_model_audio, batch_item, args.sampling_steps, args.sampling_noise)
					top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

					for i in range(batch_index.shape[0]):
						for j in range(indices_[i].shape[0]): 
							u_list_audio.append(int(batch_index[i].cpu().numpy()))
							i_list_audio.append(int(indices_[i][j].cpu().numpy()))
							edge_list_audio.append(1.0)

			# image
			u_list_image = np.array(u_list_image)
			i_list_image = np.array(i_list_image)
			edge_list_image = np.array(edge_list_image)
			self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
			self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)

			# text
			u_list_text = np.array(u_list_text)
			i_list_text = np.array(i_list_text)
			edge_list_text = np.array(edge_list_text)
			self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
			self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)

			if args.data == 'tiktok':
				# audio
				u_list_audio = np.array(u_list_audio)
				i_list_audio = np.array(i_list_audio)
				edge_list_audio = np.array(edge_list_audio)
				self.audio_UI_matrix = self.buildUIMatrix(u_list_audio, i_list_audio, edge_list_audio)
				self.audio_UI_matrix = self.model.edgeDropper(self.audio_UI_matrix)

		log('UI matrix built!')


		# GCN 和对比学习
		for i, tem in enumerate(trnLoader):

			'''
				ancs: tensor([  10, 4153,  334,  ...,  117,  125,   24], device='cuda:0')    user id
				poss: tensor([3550,  521,   96,  ..., 2457, 2226, 4632], device='cuda:0')    item id
				negs: tensor([4984,  666, 2158,  ..., 6698, 5829, 4554], device='cuda:0')    neg item id
				ancs.shape: torch.Size([1024]) poss.shape: torch.Size([1024]) negs.shape: torch.Size([1024])

				self.handler.torchBiAdj:  生成(user+item, user+item)稀疏邻接矩阵 torchBiAdj.shape: torch.Size([16018, 16018])
					tensor(indices=tensor([[    0, 10193, 10695,  ..., 16015, 16016, 16017],
										[    0,     0,     0,  ..., 16015, 16016, 16017]]),
						values=tensor([0.2500, 0.1443, 0.0606,  ..., 1.0000, 1.0000, 1.0000]),
						device='cuda:0', size=(16018, 16018), nnz=135100, layout=torch.sparse_coo)
						self.handler.torchBiAdj.shape: torch.Size([16018, 16018])
			
				self.image_UI_matrix: tensor(indices=tensor([[14014, 14666, 10909,  ..., 16015, 16016, 16017],
									[    0,     1,     2,  ..., 16015, 16016, 16017]]),
					values=tensor([0.0311, 0.0766, 0.7071,  ..., 2.0000, 2.0000, 2.0000]),
					device='cuda:0', size=(16018, 16018), nnz=17221, layout=torch.sparse_coo)
				self.image_UI_matrix.shape: torch.Size([16018, 16018])
				self.text_UI_matrix: tensor(indices=tensor([[    0, 10024,     1,  ..., 16006, 16010, 16012],
									[    0,     0,     1,  ..., 16006, 16010, 16012]]),
					values=tensor([1.0000, 0.0504, 1.0000,  ..., 2.0000, 2.0000, 2.0000]),
					device='cuda:0', size=(16018, 16018), nnz=17359, layout=torch.sparse_coo)
				self.text_UI_matrix.shape: torch.Size([16018, 16018])
				self.audio_UI_matrix: tensor(indices=tensor([[15021,     1,  9777,  ..., 16014, 16015, 16017],
									[    0,     1,     1,  ..., 16014, 16015, 16017]]),
					values=tensor([0.6325, 1.0000, 0.0327,  ..., 2.0000, 2.0000, 2.0000]),
					device='cuda:0', size=(16018, 16018), nnz=17177, layout=torch.sparse_coo)
				self.audio_UI_matrix.shape: torch.Size([16018, 16018])
			'''
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()

			self.opt.zero_grad()

			if args.data == 'tiktok':
				# diffusion_ui_adj = self.image_UI_matrix  + self.text_UI_matrix +  self.audio_UI_matrix # TODO 这里是暂时这样写的
				diffusion_ui_adj = self.image_UI_matrix
				# print("self.image_UI_matrix:", self.image_UI_matrix)
				# print("self.text_UI_matrix", self.text_UI_matrix)
				# print("self.audio_UI_matrix:", self.audio_UI_matrix)
				# print("diffusion_ui_adj:", diffusion_ui_adj)
				# all_embeddings_users, all_embeddings_items, side_embedding, content_embedding
				usrEmbeds, itmEmbeds, side_Embeds, content_Emebeds = self.model.forward(self.handler.R, self.handler.torchBiAdj, diffusion_ui_adj, self.image_II_matrix, self.text_II_matrix, self.audio_II_matrix, self.modal_fusion_II_matrix) 
				#usrEmbeds, itmEmbeds = self.model.forward(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix) # GCN
			else:
				# usrEmbeds, itmEmbeds = self.model.forward(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
				diffusion_ui_adj = self.image_UI_matrix  + self.text_UI_matrix  # TODO 这里是暂时这样写的
				self.audio_II_matrix = None
				usrEmbeds, itmEmbeds, side_Embeds, content_Emebeds = self.model.forward(self.handler.R, self.handler.torchBiAdj, diffusion_ui_adj, self.image_II_matrix, self.text_II_matrix,self.audio_II_matrix, self.modal_fusion_II_matrix)

			# Caculate Loss
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]

			# bprLoss, embLoss, regLoss = self.model.bpr_loss(ancEmbeds, posEmbeds, negEmbeds)
			bprLoss, _,  regLoss = self.model.bpr_loss(ancEmbeds, posEmbeds, negEmbeds)

			# scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			# bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
			# regLoss = self.model.reg_loss() * args.reg

			# loss = bprLoss + embLoss + regLoss
			loss = bprLoss + regLoss
			#loss = bprLoss + embLoss
			
			epRecLoss += bprLoss.item()
			epLoss += loss.item()

			# 计算对比损失
			# 剥离出用户侧，物品测的 embedding
			side_embeds_users, side_embeds_items = torch.split(side_Embeds, [args.user, args.item], dim=0)
			content_embeds_user, content_embeds_items = torch.split(content_Emebeds, [args.user, args.item], dim=0)

			# 物品-物品的对比损失
			clLoss1 = self.model.infoNCE_loss(side_embeds_items[poss], content_embeds_items[poss], args.temp) +  self.model.infoNCE_loss(side_embeds_users[ancs], content_embeds_user[ancs], args.temp) 
			# 用户-物品的对比损失
			clLoss2 = self.model.infoNCE_loss(usrEmbeds[ancs], content_embeds_items[poss], args.temp) +  self.model.infoNCE_loss(usrEmbeds[ancs], side_embeds_items[poss], args.temp) 
			clLoss = clLoss1 * args.ssl_reg + clLoss2 * args.ssl_reg

			# if args.data == 'tiktok':
			# 	usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2, usrEmbeds3, itmEmbeds3 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			# else:
			# 	usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			# if args.data == 'tiktok':
			# 	clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg
			# 	clLoss += (contrastLoss(usrEmbeds1, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds3, poss, args.temp)) * args.ssl_reg
			# 	clLoss += (contrastLoss(usrEmbeds2, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds2, itmEmbeds3, poss, args.temp)) * args.ssl_reg
			# else:
			# 	clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg

			# clLoss1 = (contrastLoss(usrEmbeds, usrEmbeds1, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds1, poss, args.temp)) * args.ssl_reg
			# clLoss2 = (contrastLoss(usrEmbeds, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds2, poss, args.temp)) * args.ssl_reg
			# if args.data == 'tiktok':
			# 	clLoss3 = (contrastLoss(usrEmbeds, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds3, poss, args.temp)) * args.ssl_reg
			# 	clLoss_ = clLoss1 + clLoss2 + clLoss3
			# else:
			# 	clLoss_ = clLoss1 + clLoss2

			# if args.cl_method == 1:
			# 	clLoss = clLoss_

			loss += clLoss

			epClLoss += clLoss.item()

			loss.backward()
			self.opt.step()

			log('Step %d/%d: bpr : %.3f ; reg : %.3f ; cl : %.3f ' % (
				i, 
				steps,
				bprLoss.item(),
        regLoss.item(),
				clLoss.item()
				), save=False, oneline=True)

		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['BPR Loss'] = epRecLoss / steps
		ret['CL loss'] = epClLoss / steps

		ret['Difusion Item-Item image loss'] = epDiLoss_image_modal / (diffusionLoader.dataset.__len__() // args.batch)
		ret['Difusion Item-Item text loss'] = epDiLoss_text_modal / (diffusionLoader.dataset.__len__() // args.batch)
		if args.data == 'tiktok':
			ret['Difusion Item-Item audio loss'] = epDiLoss_audio_modal / (diffusionLoader.dataset.__len__() // args.batch)

		# ret['Di image loss'] = epDiLoss_image / (diffusionLoader.dataset.__len__() // args.batch)
		# ret['Di text loss'] = epDiLoss_text / (diffusionLoader.dataset.__len__() // args.batch)
		# if args.data == 'tiktok':
		# 	ret['Di audio loss'] = epDiLoss_audio / (diffusionLoader.dataset.__len__() // args.batch)
		return ret

	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		epRecall, epNdcg, epPrecision = [0] * 3
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat

		if args.data == 'tiktok':
			#usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			# diffusion_ui_adj = self.image_UI_matrix  + self.text_UI_matrix +  self.audio_UI_matrix # TODO 这里是暂时这样写的
			diffusion_ui_adj = self.image_UI_matrix
			# all_embeddings_users, all_embeddings_items, side_embedding, content_embedding
			usrEmbeds, itmEmbeds, side_Embeds, content_Emebeds = self.model.forward(self.handler.R, self.handler.torchBiAdj, diffusion_ui_adj, self.image_II_matrix, self.text_II_matrix, self.audio_II_matrix, self.modal_fusion_II_matrix) 
		else:
			#usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			diffusion_ui_adj = self.image_UI_matrix  + self.text_UI_matrix  # TODO 这里是暂时这样写的
			usrEmbeds, itmEmbeds, side_Embeds, content_Emebeds = self.model.forward(self.handler.R, self.handler.torchBiAdj, diffusion_ui_adj, self.image_II_matrix, self.text_II_matrix, self.modal_fusion_II_matrix)

		# Inference
		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = torch.topk(allPreds, args.topk)
			recall, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
			epRecall += recall
			epNdcg += ndcg
			epPrecision += precision
			log('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f   ' % (i, steps, recall, ndcg, precision), save=False, oneline=True)
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		ret['Precision'] = epPrecision / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = allPrecision = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = precision = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
					precision += 1
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			precision = precision / args.topk
			allRecall += recall
			allNdcg += ndcg
			allPrecision += precision
		return allRecall, allNdcg, allPrecision

def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
	seed_it(args.seed)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	coach = Coach(handler)
	coach.run()