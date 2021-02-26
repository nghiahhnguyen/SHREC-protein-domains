import torch
import gc
import torch_geometric.data as tgd
import torch_geometric.transforms as tgt
from torch.nn import functional as F
import random
import argparse
import configparser

from dataset.protein import InMemoryProteinSurfaceDataset, ProteinSurfaceDataset
from models.models import GNN
from models.pointnet import PointNet
from models.edge_conv import SimpleEdgeConvModel, EdgeConvModel


@torch.no_grad()
def test(model, loader, args):
    model.eval()
    correct = 0
    loss = 0.
    criterion = torch.nn.CrossEntropyLoss()
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.argmax(dim=1)
        batch_loss = criterion(out, data.y)
        correct += int((pred == data.y).sum())
        # loss += F.nll_loss(out,data.y,reduction='sum').item()
        loss += batch_loss
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
    parser.add_argument('--nhid', type=int, default=256,
                        help='hidden size')
    parser.add_argument('--pooling-ratio', type=float, default=0.5,
                        help='pooling ratio')
    parser.add_argument('--dropout-ratio', type=float, default=0.5,
                        help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=100000,
                        help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=500,
                        help='patience for earlystopping')
    parser.add_argument('--num-examples', type=int, default=-1,
                        help='number of examples, all examples by default')
    parser.add_argument('--meshes-to-points', type=int, default=0,
                        help='convert the initial meshes to points cloud')
    parser.add_argument('--face-to-edge', type=int, default=1,
                        help='convert the faces to edge index')
    parser.add_argument('--model', default="gnn",
                        help='main model')
    parser.add_argument('--layer', default="gnn",
                        help='layer to use if you are using simple_edge_conv or edge_conv')
    parser.add_argument('--set-x', default=1, type=int,
                        help='set x features during data processing')
    parser.add_argument("--num-instances", type=int, default=-1,
                        help="Number of instances per class")
    parser.add_argument("--num-sample-points", type=int, default=-1,
                        help="Number of points to sample when convert from meshes to points cloud")
    parser.add_argument("--load-latest", action="store_true",
                        help="Load the latest checkpoint")
    parser.add_argument("--num-classes", type=int, default=144,
                        help="Number of classes")
    parser.add_argument("--random-rotate", action="store_true",
                        help="Use random rotate for data augmentation")
    parser.add_argument("--k", type=int, default=16,
                        help="Number of nearest neighbors for constructing knn graph")
    parser.add_argument("--in-memory-dataset", action="store_true",
                        help="Load the whole dataset into memory (faster but use more memory)")
    parser.add_argument('--use-txt', action="store_true",
                        help='whether to use physicochemical information')

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
    configuration = "off"
    if args.use_txt:
        configuration += "-txt"

    list_examples = None
    with open(classes_path, "r") as f:
        f.readline() # ignore 1st line
        num_classes, total_num_examples = map(int, f.readline().split())
        list_examples = []
        count_total_num_examples = 0
        count_num_examples = [0 for _ in range(num_classes)]
        examples = [[] for _ in range(num_classes)]
        for class_idx in range(num_classes):
            f.readline() # ignore blank line
            num_examples = int(f.readline().split()[2])
            for j in range(num_examples):
                example = int(f.readline())
                examples[class_idx].append(example)
                count_total_num_examples += 1
                count_num_examples[class_idx] += 1
        
        # print(f"The number of examples per class:")
        num_examples = []
        for i in range(num_classes):
            # print(i, count_num_examples[i])
            num_examples.append(count_num_examples[i])
        num_examples.sort()
        if args.num_instances == -1:
            median_num_examples = num_examples[len(num_examples) // 2]
            num_instances = median_num_examples
        else:
            num_instances = args.num_instances
        print(f"The number of examples per class: {num_instances}")
        
        for class_idx in range(num_classes):
            if len(examples[class_idx]) >= num_instances:
                population = examples[class_idx][:]
                random.shuffle(population)
                population = population[:num_instances]
            else:
                population = []
                for _ in range(num_instances):
                    dice = random.randint(0, len(examples[class_idx]) - 1)
                    population.append(examples[class_idx][dice])
            for example in population:
                list_examples.append((example, class_idx))

        test_ratio = 0.15 #@param {type:"number"}
        val_ratio = 0.15 #@param {type:"number"} 
        random.shuffle(list_examples)
        if args.num_examples == -1:
            args.num_examples = len(list_examples)
        print(f"The number of original examples (after oversampling and undersampling): {len(list_examples)}")
        print(f"The number of used examples: {args.num_examples}")
        list_examples = list_examples[:args.num_examples]
        list_examples_test = list_examples[:int(test_ratio * len(list_examples))]
        list_examples_val = list_examples[int(test_ratio * len(list_examples))+1:int((test_ratio+val_ratio)*len(list_examples))]
        list_examples_train = list_examples[int((test_ratio+val_ratio)*len(list_examples))+1:]
        assert(total_num_examples == count_total_num_examples)  

    list_transforms = []
    random_rotate = tgt.Compose([
        tgt.RandomRotate(degrees=180, axis=0),
        tgt.RandomRotate(degrees=180, axis=1),
        tgt.RandomRotate(degrees=180, axis=2),
    ])
    if args.random_rotate:
        list_transforms.append(random_rotate)
    if args.face_to_edge == 1:
        list_transforms.append(tgt.FaceToEdge(True))
    if args.meshes_to_points == 1:
        list_transforms.append(tgt.SamplePoints(num=args.num_sample_points))
    transforms = tgt.Compose(list_transforms)

    if args.in_memory_dataset:
        DatasetType = InMemoryProteinSurfaceDataset
    else:
        DatasetType = ProteinSurfaceDataset

    dataset_path = f"data/num-instances={args.num_instances}-num-sample-points={args.num_sample_points}-use-txt={args.use_txt}-set-x={args.set_x}"
    print(f"Dataset path: {dataset_path}")

    train_off_dataset = DatasetType(dataset_path, list_examples_train, off_train_folder_path, txt_train_folder_path, args, "train", transform=transforms)
    val_off_dataset = DatasetType(dataset_path, list_examples_val, off_train_folder_path, txt_train_folder_path, args, "val", transform=transforms)
    test_off_dataset = DatasetType(dataset_path, list_examples_test, off_train_folder_path, txt_train_folder_path, args, "test", transform=transforms)
    train_off_loader = tgd.DataLoader(train_off_dataset, batch_size=args.batch_size, shuffle=True)
    val_off_loader = tgd.DataLoader(val_off_dataset, batch_size=args.batch_size, shuffle=True)
    test_off_loader = tgd.DataLoader(test_off_dataset, batch_size=args.batch_size, shuffle=True)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.num_features = train_off_dataset.num_node_features
    # args.num_classes = int(train_off_dataset.num_classes)
    if args.use_txt:
        args.num_features = 6
    else:
        args.num_features = 3
    print("Number of features dimension:", args.num_features)
    print("Number of classes:", args.num_classes)

    print(args)

    if args.model == "pointnet":
        model = PointNet(args).to(args.device)
    elif args.model == "simple_edge_conv":
        model = SimpleEdgeConvModel(args).to(args.device)
    elif args.model == "edge_conv":
        model = EdgeConvModel(args).to(args.device)
    else:
        model = GNN(args).to(args.device)
    
    # print(model)
    
    model_save_path = f'{args.model}-{configuration}-latest.pth'

    if args.load_latest:
        model.load_state_dict(torch.load(model_save_path))
        val_acc, val_loss = test(model, val_off_loader, args)
        print("Validation loss: {}\taccuracy:{}".format(val_loss, val_acc))
        torch.save(model.state_dict(), model_save_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    # torch.cuda.empty_cache()
    # gc.collect()


    min_loss = 1e10
    patience = 0
    epoch = 0
    for epoch in range(args.epochs):
        model.train()
        training_loss = 0
        for i, data in enumerate(train_off_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out = model(data)
            target = data.y.long()
            loss = criterion(out, target)
            training_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        training_loss /= len(train_off_loader.dataset)
        print("Training loss: {}".format(training_loss))
        val_acc, val_loss = test(model, val_off_loader, args)
        print("Validation loss: {}\taccuracy:{}".format(val_loss, val_acc))
        if val_loss < min_loss:
            torch.save(model.state_dict(), model_save_path)
            print("Model saved at epoch{}".format(epoch))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break 

    if epoch:
        print("Last epoch before stopping:", epoch)

    test_acc, test_loss = test(model, test_off_loader, args)
    print("Test loss:{}\taccuracy:{}".format(test_loss, test_acc))

if __name__ == "__main__":
    main()