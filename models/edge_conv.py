import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool, EdgeConv, DynamicEdgeConv
from torch.nn import Linear


class SimpleEdgeConvModel(torch.nn.Module):
	def __init__(self, args):
		super(SimpleEdgeConvModel, self).__init__()

		torch.manual_seed(args.seed)
		if args.layer == "edge_conv":
			layer = EdgeConv
		else:
			layer = DynamicEdgeConv
		self.conv1 = Linear(args.num_features * 2, args.nhid)
		self.edge_conv1 = layer(self.conv1)
		self.conv2 = Linear(args.nhid * 2, args.nhid)
		self.edge_conv2 = layer(self.conv2)
		self.classifier = Linear(args.nhid * 2, args.num_classes)

	def forward(self, data):
		pos, batch = data.pos, data.batch
		# Compute the kNN graph:
		# Here, we need to pass the batch vector to the function call in order
		# to prevent creating edges between points of different examples.
		# We also add `loop=True` which will add self-loops to the graph in
		# order to preserve central point information.
		edge_index = knn_graph(pos, k=16, batch=batch, loop=True)

		# 3. Start bipartite message passing.
		h = self.edge_conv1(h=pos, pos=pos, edge_index=edge_index)
		h = h.relu()
		h = self.edge_conv2(h=h, pos=pos, edge_index=edge_index)
		h = h.relu()

		# 4. Global Pooling.
		h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

		# 5. Classifier.
		return self.classifier(h)
