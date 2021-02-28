# SHREC-protein-domains

## Installation

Our method use the PyTorch Geometric package. We provide instructions to install the package below. Alternatively, a more elaborated installation instruction can be found on the package [documents page](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

Open the file ```install_torch_geometric.sh``` file. Modify the value ```1.7.0``` and ```cu110``` on the first line by PyTorch version (1.4.0, 1.5.0, 1.6.0, 1.7.0) (You should round your PyTorch version to the second version number e.g., if your PyTorchVersion is 1.6.3, you should replace 1.7.0 in the original file with 1.6.0) and your specific CUDA version (cpu, cu92, cu101, cu102, cu110).

Run

```bash
bash install_torch_geometric.sh
```

## Run code

The code can be run from the file ```main.py```. A sample command can be found in ```run.sh```. You can run the sample command by running:

```bash
bash run.sh
```

If you want to know the meaning of the arguments, you can find more details in the file ```main.py```
