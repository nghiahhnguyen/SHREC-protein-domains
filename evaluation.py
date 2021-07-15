from utils.metrics import *
import os.path as osp
from glob import glob
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_labeled', action='store_true')
    parser.add_argument('--method', type=str)
    parser.add_argument('--electrostatic-track', action='store_true')
    args = parser.parse_args()


    if args.method == 'ensemble':
        methods = ['pointnet-off-128-256-10', 'edge_conv-off-512-256-10']
        dist_arr_list = []
        prob_arr_list = []
        for method in methods:
            emb_dir = f'saved_{"test_labeled" if args.is_labeled else "unlabeled"}_embeddings/{method}'
            # int(osp.basename(f).split('.')[0]), 
            file_list = [(int(osp.basename(f).split('.')[0]),f) for f in glob(emb_dir + '/*', recursive=True) if not osp.isdir(f) and not osp.basename(f).split('_')[-1] == 'prob.npy']
            file_list = sorted(file_list)
            file_list = [f for _, f in file_list]
            label_arr, dist_arr = label_distance_matrix_from_file(file_list, file_list)
            _, dist_arr = np.dsplit(dist_arr, 2)
            dist_arr = dist_arr.squeeze()
            dist_arr_list.append(dist_arr)

            file_list = [(int(osp.basename(f).split('.')[0].split('_')[0]),f) for f in glob(emb_dir + '/*', recursive=True) if not osp.isdir(f) and osp.basename(f).split('_')[-1] == 'prob.npy']
            file_list = sorted(file_list)
            file_list = [f for _, f in file_list]
            prob = np.array([np.load(f) for f in file_list])
            prob_arr_list.append(prob)

        dist_arr = np.average(np.array(dist_arr_list), axis=0, weights=[0.4, 0.6])
        label_arr = np.argmax(np.average(np.array(prob_arr_list), axis=0, weights=[0.4, 0.6]), axis=-1)
        
    else:

        emb_dir = f'saved_{"test_labeled" if args.is_labeled else "unlabeled"}_embeddings/{args.method}'
        # int(osp.basename(f).split('.')[0]), 
        file_list = [(int(osp.basename(f).split('.')[0]),f) for f in glob(emb_dir + '/*', recursive=True) if not osp.isdir(f) and not osp.basename(f).split('_')[-1] == 'prob.npy']
        file_list = sorted(file_list)
        file_list = [f for _, f in file_list]
        label_arr, dist_arr = label_distance_matrix_from_file(file_list, file_list)

        _, dist_arr = np.dsplit(dist_arr, 2)
        dist_arr = dist_arr.squeeze()

    if not args.electrostatic-track:
        with open(f'{args.method}-classification-submission.txt', 'w') as f:
            for label in label_arr:
                print(label+1, file=f) 

        df = pd.DataFrame(data=dist_arr[0:, 0:],
                        index=[i for i in range(dist_arr.shape[0])],
                        columns=['' for i in range(dist_arr.shape[1])])

        df.to_csv(f'{args.method}-retrieval-submission.txt', index=False, header=False, sep=' ')
    
    else:
        dist_arr = dist_arr.flatten()
        np.save('{args.method}-distance-matrix-submission.matrix', dist_arr, allow_pickle=False)

    if args.is_labeled:
        pf = retrieval_success(dist_arr, label_arr, 1)
        ps = retrieval_success(dist_arr, label_arr, 2)

        print(pf)
        print(ps)
        
