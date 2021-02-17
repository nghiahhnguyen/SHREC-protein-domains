import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric import io as tgio

class InMemoryProteinSurfaceDataset(InMemoryDataset):
    def __init__(self, root, list_examples, final=False, use_txt=False, transform=None, pre_transform=None):
        self.list_examples = list_examples
        self.use_txt = use_txt
        self.final = final
        super(InMemoryProteinSurfaceDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        if self.final == False:
            off_folder_path = off_train_folder_path
            txt_folder_path = txt_train_folder_path
        else:
            off_folder_path = off_test_folder_path
            txt_folder_path = txt_test_folder_path
        
        data_list = []
        for example_idx, class_idx in self.list_examples:
            off_path =  f"{off_folder_path}/{example_idx}.off" 
            txt_path =  f"{txt_folder_path}/{example_idx}.txt"
            protein = read_off(off_path)
            protein.x = protein.pos
            protein.y = torch.Tensor([class_idx]).type(torch.LongTensor)
            if self.use_txt:
                txt_data = tgio.read_txt_array(txt_path)
                torch.x = torch.cat((protein.x, txt_data), 1)
            data_list.append(protein)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 
