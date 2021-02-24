import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool, EdgeConv, DynamicEdgeConv
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
		self.edge_conv1 = layer(self.conv1, k=args.k)
		self.conv2 = nn.Linear(args.nhid * 2, args.nhid)
		self.edge_conv2 = layer(self.conv2, k=args.k)
		self.classifier = nn.Linear(args.nhid * 2, args.num_classes)
		self.args = args

	def forward(self, data):
		pos, batch = data.pos, data.batch
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
		# self.conv1 = Linear(args.num_features * 2, args.nhid)
		# self.edge_conv1 = layer(self.conv1)
		# self.conv2 = Linear(args.nhid * 2, args.nhid)
		# self.edge_conv2 = layer(self.conv2)
		# self.classifier = Linear(args.nhid * 2, args.num_classes)

		self.args = args

		self.conv1 = nn.Sequential
		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(64)
		self.bn3 = nn.BatchNorm2d(128)
		self.bn4 = nn.BatchNorm2d(256)
		self.bn5 = nn.BatchNorm1d(args.nhid)

		self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
								self.bn1,
								nn.LeakyReLU(negative_slope=0.2))
		self.edge_conv1 = layer(self.conv1, k=args.k)
		self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
								self.bn2,
								nn.LeakyReLU(negative_slope=0.2))
		self.edge_conv2 = layer(self.conv2, k=args.k)
		self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
								self.bn3,
								nn.LeakyReLU(negative_slope=0.2))
		self.edge_conv3 = layer(self.conv3, k=args.k)
		self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
								self.bn4,
								nn.LeakyReLU(negative_slope=0.2))
		self.edge_conv4 = layer(self.conv4, k=args.k)
		self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
								self.bn5,
								nn.LeakyReLU(negative_slope=0.2))
		self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
		self.bn6 = nn.BatchNorm1d(512)
		self.dp1 = nn.Dropout(p=args.dropout)
		self.linear2 = nn.Linear(512, 256)
		self.bn7 = nn.BatchNorm1d(256)
		self.dp2 = nn.Dropout(p=args.dropout)
		self.linear3 = nn.Linear(256, args.num_classes)

	def forward(self, data):
		pos, batch = data.pos, data.batch
		edge_index = knn_graph(pos, k=self.args.k, batch=batch, loop=True)
		batch_size = data.num_graphs

		if self.args.layer == "edge_conv":
			edge_info = edge_index
		else:
			edge_info = batch

		x1 = self.edge_conv1(pos, edge_info)
		x2 = self.edge_conv2(x1, edge_info)
		x3 = self.edge_conv3(x2, edge_info)
		x4 = self.edge_conv4(x3, edge_info)
		x = torch.cat((x1, x2, x3, x4), dim=1)

		x = self.conv5(x)
		x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
		x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
		x = torch.cat((x1, x2), 1)

		x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
		x = self.dp1(x)
		x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
		x = self.dp2(x)
		x = self.linear3(x)
		return x