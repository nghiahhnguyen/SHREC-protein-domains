import torch
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric import io as tgio
import os.path as osp

from .io import read_off


class InMemoryProteinSurfaceDataset(InMemoryDataset):
    def __init__(self, root, list_examples, off_folder_path, txt_folder_path, args, final=False, use_txt=False, transform=None, pre_transform=None):
        self.list_examples = list_examples
        self.use_txt = use_txt
        self.final = final
        self.off_folder_path = off_folder_path
        self.txt_folder_path = txt_folder_path
        self.set_x = args.set_x
        super(InMemoryProteinSurfaceDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        # if self.final == False:
        #     off_folder_path = off_train_folder_path
        #     txt_folder_path = txt_train_folder_path
        # else:
        #     off_folder_path = off_test_folder_path
        #     txt_folder_path = txt_test_folder_path
        
        data_list = []
        for example_idx, class_idx in self.list_examples:
            off_path =  f"{self.off_folder_path}/{example_idx}.off" 
            txt_path =  f"{self.txt_folder_path}/{example_idx}.txt"
            protein = read_off(off_path)
            if self.set_x == 1:
                protein.x = protein.pos
            protein.y = torch.Tensor([class_idx]).type(torch.LongTensor)
            if self.use_txt:
                txt_data = tgio.read_txt_array(txt_path)
                protein.x = torch.cat((protein.pos, txt_data), 1)
            data_list.append(protein)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 

class ProteinSurfaceDataset(Dataset):
    def __init__(self, root, list_examples, off_folder_path, txt_folder_path, args, split, final=False, transform=None, pre_transform=None):
        self.list_examples = list_examples
        self.use_txt = args.use_txt
        self.final = final
        self.off_folder_path = off_folder_path
        self.txt_folder_path = txt_folder_path
        self.set_x = args.set_x
        self.args = args
        self.split = split
        super(ProteinSurfaceDataset, self).__init__(root, transform, pre_transform)
    
    @property
    def processed_file_names(self):
        ret = []
        for idx in range(len(self.list_examples)):
            ret.append(f"data_{self.split}_{idx}.pt")
        return ret

    def process(self):
        for idx, (example_idx, class_idx) in enumerate(self.list_examples):
            off_path =  f"{self.off_folder_path}/{example_idx}.off" 
            txt_path =  f"{self.txt_folder_path}/{example_idx}.txt"
            protein = read_off(off_path)
            if self.set_x == 1:
                protein.x = protein.pos
            protein.y = torch.Tensor([class_idx]).type(torch.LongTensor)
            if self.use_txt:
                txt_data = tgio.read_txt_array(txt_path)
                protein.x = torch.cat((protein.pos, txt_data), 1)
            torch.save(protein, osp.join(self.processed_dir, f"data_{self.split}_{idx}.pt"))
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{self.split}_{idx}.pt"))
        return data
    