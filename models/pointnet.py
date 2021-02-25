import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool
from torch.nn import Linear

from .layers.pointnet import PointNetLayer


class PointNet(torch.nn.Module):
	def __init__(self, args):
		super(PointNet, self).__init__()

		torch.manual_seed(args.seed)
		self.conv1 = PointNetLayer(args.num_features, args.nhid)
		self.conv2 = PointNetLayer(args.nhid, args.nhid)
		self.classifier = Linear(args.nhid, args.num_classes)

	def forward(self, data):
		pos, batch = data.pos, data.batch
		# Compute the kNN graph:
		# Here, we need to pass the batch vector to the function call in order
		# to prevent creating edges between points of different examples.
		# We also add `loop=True` which will add self-loops to the graph in
		# order to preserve central point information.
		edge_index = knn_graph(pos, k=16, batch=batch, loop=True)

		# 3. Start bipartite message passing.
		h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
		h = h.relu()
		h = self.conv2(h=h, pos=pos, edge_index=edge_index)
		h = h.relu()

		# 4. Global Pooling.
		h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

		# 5. Classifier.
		return self.classifier(h)

	
	def encode(self, data):
		pos, batch = data.pos, data.batch
		# Compute the kNN graph:
		# Here, we need to pass the batch vector to the function call in order
		# to prevent creating edges between points of different examples.
		# We also add `loop=True` which will add self-loops to the graph in
		# order to preserve central point information.
		edge_index = knn_graph(pos, k=16, batch=batch, loop=True)

		# 3. Start bipartite message passing.
		h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
		h = h.relu()
		h = self.conv2(h=h, pos=pos, edge_index=edge_index)
		h = h.relu()

		# 4. Global Pooling.
		h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

		return h