import torch_geometric.data as tgd
import torch_geometric.transforms as tgt
from sys import stdin
import random
import argparse

random.seed(167)

classes_path = "/content/data/classTraining.cla"

list_examples = None
with open(classes_path, "r") as f:
    f.readline() # ignore 1st line
    num_classes, total_num_examples = map(int, f.readline().split())
    # list_examples = [[] for _ in range(num_classes)]
    list_examples = []
    count_total_num_examples = 0
    for class_idx in range(num_classes):
        f.readline() # ignore blank line
        num_examples = int(f.readline().split()[2])
        for j in range(num_examples):
            example = int(f.readline())
            # list_examples[class_idx].append(example)
            list_examples.append((example, class_idx))
            count_total_num_examples += 1
    # list_examples = [l[:15] for l in list_examples]
    test_ratio = 0.15 #@param {type:"number"}
    val_ratio = 0.15 #@param {type:"number"} 
    random.shuffle(list_examples)
    print(len(list_examples))
    list_examples = list_examples[:2000]
    list_examples_test = list_examples[:int(test_ratio * len(list_examples))]
    list_examples_val = list_examples[int(test_ratio * len(list_examples))+1:int((test_ratio+val_ratio)*len(list_examples))]
    list_examples_train = list_examples[int((test_ratio+val_ratio)*len(list_examples))+1:]
    assert(total_num_examples == count_total_num_examples)  

train_off_dataset = InMemoryProteinSurfaceDataset("/content/data", list_examples_train, transform=tgt.FaceToEdge(True))
val_off_dataset = ProteinSurfaceDataset("data", list_examples_val, transform=tgt.FaceToEdge(True))
test_off_dataset = ProteinSurfaceDataset("data", list_examples_test, transform=tgt.FaceToEdge(True))
train_off_loader = tgd.DataLoader(train_off_dataset, batch_size=8, shuffle=True)
val_off_loader = tgd.DataLoader(val_off_dataset, batch_size=8, shuffle=True)
test_off_loader = tgd.DataLoader(test_off_dataset, batch_size=8, shuffle=True)

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=100000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')

args = parser.parse_args("")
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.num_features = train_off_dataset.num_node_features
args.num_classes = int(train_off_dataset.num_classes)

model = Net(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)

def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        data.y = data.y.long()
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset), loss / len(loader.dataset)

min_loss = 1e10
patience = 0
for epoch in range(args.epochs):
    model.train()
    training_loss = 0
    for i, data in enumerate(train_off_loader):
        data = data.to(args.device)
        out = model(data)
        target = data.y.long()
        loss = F.nll_loss(out, target)
        training_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    training_loss /= len(train_off_loader)
    print("Training loss:{}".format(training_loss))
    train_acc, train_loss = test(model, train_off_loader)
    print("Training loss:{}\taccuracy:{}".format(train_loss, train_acc))
    val_acc, val_loss = test(model, test_off_loader)
    print("Validation loss:{}\taccuracy:{}".format(val_loss, val_acc))
    if val_loss < min_loss:
        torch.save(model.state_dict(),'latest.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break 

test(model, test_off_loader)
