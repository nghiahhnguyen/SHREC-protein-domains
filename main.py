import torch
import torch_geometric.data as tgd
import torch_geometric.transforms as tgt
from torch.nn import functional as F
import random
import argparse
import configparser

from dataset.in_memory import InMemoryProteinSurfaceDataset
from models.models import GNN
from models.pointnet import PointNet


def test(model, loader, args):
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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=167,
                        help='seed')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--nhid', type=int, default=128,
                        help='hidden size')
    parser.add_argument('--pooling-ratio', type=float, default=0.5,
                        help='pooling ratio')
    parser.add_argument('--dropout-ratio', type=float, default=0.5,
                        help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=100000,
                        help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for earlystopping')
    parser.add_argument('--num-examples', type=int, default=3585,
                        help='number of examples, all examples by default')
    parser.add_argument('--meshes-to-points', type=int, default=0,
                        help='convert the initial meshes to points cloud')
    parser.add_argument('--face-to-edge', type=int, default=1,
                        help='convert the faces to edge index')
    parser.add_argument('--model', default="gnn",
                        help='main model')

    args = parser.parse_args()
    random.seed(args.seed)
    config = configparser.ConfigParser()
    config.read("config.ini")
    config_paths = config["PATHS"]
    base_path = config_paths["base_path"]
    classes_path = base_path + config_paths["classes_path"]
    off_train_folder_path = base_path + config_paths["off_train_folder_path"]
    txt_train_folder_path = base_path + config_paths["txt_train_folder_path"]
    off_final_test_folder_path = base_path + config_paths["off_final_test_folder_path"]
    txt_final_test_folder_path = base_path + config_paths["txt_final_test_folder_path"]

    list_examples = None
    with open(classes_path, "r") as f:
        f.readline() # ignore 1st line
        num_classes, total_num_examples = map(int, f.readline().split())
        list_examples = []
        count_total_num_examples = 0
        for class_idx in range(num_classes):
            f.readline() # ignore blank line
            num_examples = int(f.readline().split()[2])
            for j in range(num_examples):
                example = int(f.readline())
                list_examples.append((example, class_idx))
                count_total_num_examples += 1
        test_ratio = 0.15 #@param {type:"number"}
        val_ratio = 0.15 #@param {type:"number"} 
        random.shuffle(list_examples)
        print(f"The number of original examples: {len(list_examples)}")
        print(f"The number of used examples: {args.num_examples}")
        list_examples = list_examples[:args.num_examples]
        list_examples_test = list_examples[:int(test_ratio * len(list_examples))]
        list_examples_val = list_examples[int(test_ratio * len(list_examples))+1:int((test_ratio+val_ratio)*len(list_examples))]
        list_examples_train = list_examples[int((test_ratio+val_ratio)*len(list_examples))+1:]
        assert(total_num_examples == count_total_num_examples)  

    list_transforms = []
    if args.face_to_edge == 1:
        list_transforms.append(tgt.FaceToEdge(True))
    if args.meshes_to_points == 1:
        list_transforms.append(tgt.SamplePoints(num=128))
    transforms = tgt.Compose(list_transforms)

    train_off_dataset = InMemoryProteinSurfaceDataset(base_path, list_examples_train, off_train_folder_path, txt_train_folder_path, transform=transforms)
    val_off_dataset = InMemoryProteinSurfaceDataset(base_path, list_examples_val, off_train_folder_path, txt_train_folder_path, transform=transforms)
    test_off_dataset = InMemoryProteinSurfaceDataset(base_path, list_examples_test, off_train_folder_path, txt_train_folder_path, transform=transforms)
    train_off_loader = tgd.DataLoader(train_off_dataset, batch_size=args.batch_size, shuffle=True)
    val_off_loader = tgd.DataLoader(val_off_dataset, batch_size=args.batch_size, shuffle=True)
    test_off_loader = tgd.DataLoader(test_off_dataset, batch_size=args.batch_size, shuffle=True)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.num_features = train_off_dataset.num_node_features
    args.num_classes = int(train_off_dataset.num_classes)

    print(args)

    if args.model == "pointnet":
        model = PointNet(args).to(args.device)
    else:
        model = GNN(args).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    min_loss = 1e10
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        training_loss = 0
        for i, data in enumerate(train_off_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out = model(data)
            target = data.y.long()
            loss = F.nll_loss(out, target)
            training_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        training_loss /= len(train_off_loader.dataset)
        print("Training loss:{}".format(training_loss))
        val_acc, val_loss = test(model, val_off_loader, args)
        print("Validation loss:{}\taccuracy:{}".format(val_loss, val_acc))
        if val_loss < min_loss:
            torch.save(model.state_dict(),f'{args.model}-latest.pth')
            print("Model saved at epoch{}".format(epoch))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break 

    test_acc, test_loss = test(model, test_off_loader, args)
    print("Test loss:{}\taccuracy:{}".format(test_loss, test_acc))

if __name__ == "__main__":
    main()