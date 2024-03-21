import pandas as pd
import os

def main():
    data_train = pd.read_csv('~/Projects/ensemble/ensemble/datasets/singlecoil_knee_train.csv')
    data_test = pd.read_csv('~/Projects/ensemble/ensemble/datasets/singlecoil_knee_test.csv')
    data_val = pd.read_csv('~/Projects/ensemble/ensemble/datasets/singlecoil_knee_val.csv')
    data_fastmriplus = pd.read_csv('~/Projects/ensemble/external/fastmri-plus/Annotations/knee.csv')

    data_all = pd.concat([data_test, data_val])

    # healthy subjects
    cases = ['train', 'val', 'test']
    for case in cases:
        data_fastmriplus_mod = data_fastmriplus
        data_fastmriplus_mod['file'] = data_fastmriplus['file'].apply(lambda x: 'knee/data/singlecoil_' + case + '/' + x + '.h5')
        data_all_train = data_train.merge(data_fastmriplus_mod.drop_duplicates(subset=['file']), left_on='filename', right_on='file', how='left', indicator=True)
        data_hs_train = data_all_train[data_all_train['_merge'] == 'left_only']
        data_hs_train.to_csv('~/Projects/ensemble/ensemble/datasets/singlecoil_knee_' + case + '_hs.csv', index=False)
        data_pat_train = data_all_train[data_all_train['_merge'] == 'both']
        data_pat_train.to_csv('~/Projects/ensemble/ensemble/datasets/singlecoil_knee_' + case + '_pat.csv', index=False)

    testlist = data_test['filename'].apply(lambda x: os.path.basename(x).split('_')[0]).values.tolist()
    vallist = data_val['filename'].apply(lambda x: os.path.basename(x).split('.')[0]).values.tolist()
    allist = data_all['filename'].apply(lambda x: os.path.basename(x).split('.')[0].split('_')[0]).values.tolist()
    #fmrilist = data_fastmriplus.values.tolist()
    #fmrilist = [x[0] for x in fmrilist]
    fmrilist = list(dict.fromkeys(data_fastmriplus['file'].values.tolist()))
    sellist = [x for x in fmrilist if x in allist]
    data_set = data_all[data_all['filename'].apply(lambda x: os.path.basename(x).split('.')[0].split('_')[0] in sellist)]
    data_set.to_csv('~/Projects/ensemble/ensemble/datasets/singlecoil_knee_fastmriplus.csv', index=False)

    # brain
    data_all = pd.read_csv('~/Projects/ensemble/ensemble/datasets/multicoil_brain_val_filtered.csv')
    data_fastmriplus = pd.read_csv('~/Projects/ensemble/external/fastmri-plus/Annotations/brain.csv')
    allist = data_all['filename'].apply(lambda x: os.path.basename(x).split('.')[0]).values.tolist()
    fmrilist = list(dict.fromkeys(data_fastmriplus['file'].values.tolist()))
    sellist = [x for x in fmrilist if x in allist]
    data_set = data_all[
        data_all['filename'].apply(lambda x: os.path.basename(x).split('.')[0].split('_')[0] in sellist)]
    data_set.to_csv('~/Projects/ensemble/ensemble/datasets/multicoil_brain_fastmriplus.csv', index=False)

if __name__ == '__main__':
    main()