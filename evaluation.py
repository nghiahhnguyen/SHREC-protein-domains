from utils.metrics import *
import os.path as osp
from glob import glob
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_labeled', action='store_true')
    parser.add_argument('--method', type=str)
    args = parser.parse_args()

    emb_dir = f'/home/nhhnghia/SHREC-protein-domains/saved_{"test_labeled" if args.is_labeled else "unlabeled"}_embeddings/{args.method}'
    # int(osp.basename(f).split('.')[0]), 
    file_list = [(int(osp.basename(f).split('.')[0]),f) for f in glob(emb_dir + '/*', recursive=True) if not osp.isdir(f)]
    file_list = sorted(file_list)
    file_list = [f for _, f in file_list]
    label_arr, dist_arr = label_distance_matrix_from_file(file_list, file_list)

    print(dist_arr)
    # with open(f'{args.method}-submission.txt')
    #     for x_list in dist_arr:
    #         for y in x_list:
    
    df = pd.DataFrame(data=dist_arr[0:, 0:],
                    index=[i for i in range(dist_arr.shape[0])],
                    columns=['' for i in range(dist_arr.shape[1])])

    df.to_csv(f'{args.method}-submission.txt', index=False, header=False, sep=' ')

    if args.is_labeled:
        # label_df = pd.read_csv('/home/nhtduy/SHREC21/protein-physicochemical/trainingClass.csv')
        # label_df = label_df.sort_values(by=['off_file'])
        # label_dict = {key: value for key, value in label_df.to_records(index=False)}
        pf = retrieval_success(dist_arr, label_arr, 1)
        ps = retrieval_success(dist_arr, label_arr, 2)

        print(pf)
        print(ps)
        
