TORCH_VERSION="torch-1.7.0+cu110"
TORCH_SOURCE="https://pytorch-geometric.com/whl/${TORCH_VERSION}.html"
echo $TORCH_SOURCE
pip install --no-index torch-scatter -f $TORCH_SOURCE
pip install --no-index torch-sparse -f $TORCH_SOURCE
pip install --no-index torch-cluster -f $TORCH_SOURCE
pip install --no-index torch-spline-conv -f $TORCH_SOURCE
pip install torch-geometric