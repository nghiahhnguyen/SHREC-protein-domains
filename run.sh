# mkdir data
# python3 main.py --face-to-edge 0 --meshes-to-points 1 --model edge_conv --layer dynamic_edge_conv --set-x 0 --lr 0.0005 --num-instances 10 --num-sample-points 512 --load-latest
python3 main.py --face-to-edge 0 --meshes-to-points 1 --model simple_edge_conv --layer dynamic_edge_conv --set-x 0 --lr 0.0005 --num-instances 10 --num-sample-points 512 --use-txt --batch-size 1
# python3 main.py --face-to-edge 0 --meshes-to-points 1 --model edge_conv --layer dynamic_edge_conv --set-x 0 --lr 0.0005 --num-instances 5 --num-sample-points 32 --in-memory-dataset
