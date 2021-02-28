import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool, global_mean_pool, EdgeConv, DynamicEdgeConv, SAGPooling, BatchNorm
from torch import nn


class SimpleEdgeConvModel(torch.nn.Module):
	def __init__(self, args):
		super(SimpleEdgeConvModel, self).__init__()

		torch.manual_seed(args.seed)
		if args.layer == "edge_conv":
			layer = EdgeConv
		else:
			layer = DynamicEdgeConv
		self.conv1 = nn.Linear(args.num_features * 2, args.nhid)
		if args.layer == "edge_conv":
			self.edge_conv1 = layer(self.conv1)
		else:
			self.edge_conv1 = layer(self.conv1, k=args.k)
		self.conv2 = nn.Linear(args.nhid * 2, args.nhid)
		if args.layer == "edge_conv":
			self.edge_conv2 = layer(self.conv2)
		else:
			self.edge_conv2 = layer(self.conv2, k=args.k)
		self.classifier = nn.Linear(args.nhid, args.num_classes)
		self.args = args

	def forward(self, data):
		pos, batch = data.pos, data.batch
		# if self.args.use_txt:
		# 	pos = data.x

		# Compute the kNN graph:
		# Here, we need to pass the batch vector to the function call in order
		# to prevent creating edges between points of different examples.
		# We also add `loop=True` which will add self-loops to the graph in
		# order to preserve central point information.
		edge_index = knn_graph(pos, k=self.args.k, batch=batch, loop=True)

		if self.args.layer == "edge_conv":
			edge_info = edge_index
		else:
			edge_info = batch

		# 3. Start bipartite message passing.
		h = self.edge_conv1(pos, edge_info)
		h = h.relu()
		h = self.edge_conv2(h, edge_info)
		h = h.relu()

		# 4. Global Pooling.
		h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

		# 5. Classifier.
		return self.classifier(h)


class EdgeConvModel(torch.nn.Module):
	def __init__(self, args):
		super(EdgeConvModel, self).__init__()

		torch.manual_seed(args.seed)
		if args.layer == "edge_conv":
			layer = EdgeConv
		else:
			layer = DynamicEdgeConv
		self.args = args

		self.conv1 = nn.Sequential
		self.bn1 = BatchNorm(64)
		self.bn2 = BatchNorm(64)
		self.bn3 = BatchNorm(128)
		self.bn4 = BatchNorm(256)
		self.bn5 = BatchNorm(args.nhid)
		self.conv1 = nn.Sequential(nn.Linear(args.num_features*2, 64),
								self.bn1,
								nn.LeakyReLU(negative_slope=0.2))
		if args.layer == "edge_conv":
			self.edge_conv1 = layer(self.conv1)
		else:
			self.edge_conv1 = layer(self.conv1, k=args.k)
		self.conv2 = nn.Sequential(nn.Linear(64*2, 64),
								self.bn2,
								nn.LeakyReLU(negative_slope=0.2))
		if args.layer == "edge_conv":
			self.edge_conv2 = layer(self.conv2)
		else:
			self.edge_conv2 = layer(self.conv2, k=args.k)
		self.conv3 = nn.Sequential(nn.Linear(64*2, 128),
								self.bn3,
								nn.LeakyReLU(negative_slope=0.2))
		if args.layer == "edge_conv":
			self.edge_conv3 = layer(self.conv3)
		else:
			self.edge_conv3 = layer(self.conv3, k=args.k)
		self.conv4 = nn.Sequential(nn.Linear(128*2, 256),
								self.bn4,
								nn.LeakyReLU(negative_slope=0.2))
		if args.layer == "edge_conv":
			self.edge_conv4 = layer(self.conv4)
		else:
			self.edge_conv4 = layer(self.conv4, k=args.k)
		self.conv5 = nn.Sequential(nn.Linear(1024, args.nhid),
								self.bn5,
								nn.LeakyReLU(negative_slope=0.2))
		if args.layer == "edge_conv":
			self.edge_conv5 = layer(self.conv5)
		else:
			self.edge_conv5 = layer(self.conv5, k=args.k)
		self.graph_pool1 = SAGPooling(args.nhid)
		self.linear1 = nn.Linear(args.nhid * 2, 512, bias=False)
		self.bn6 = nn.BatchNorm1d(512)
		self.dp1 = nn.Dropout(p=args.dropout_ratio)
		self.linear2 = nn.Linear(512, 256)
		self.bn7 = nn.BatchNorm1d(256)
		self.dp2 = nn.Dropout(p=args.dropout_ratio)
		self.linear3 = nn.Linear(256, args.num_classes)

	def forward(self, data):
		pos, batch = data.pos, data.batch
		# if self.args.use_txt:
		# 	pos = data.x
		edge_index = knn_graph(pos, k=self.args.k, batch=batch, loop=True)

		if self.args.layer == "edge_conv":
			edge_info = edge_index
		else:
			edge_info = batch

		x1 = self.edge_conv1(pos, edge_info)
		x2 = self.edge_conv2(x1, edge_info)
		x3 = self.edge_conv3(x2, edge_info)
		x4 = self.edge_conv4(x3, edge_info)
		x = torch.cat((x1, x2, x3, x4), dim=1)
		x = self.edge_conv5(x, edge_info)
		x1 = global_max_pool(x, batch)
		x2 = global_mean_pool(x, batch)
		x = torch.cat((x1, x2), 1)

		x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
		x = self.dp1(x)
		x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
		x = self.dp2(x)
		x = self.linear3(x)
		return x