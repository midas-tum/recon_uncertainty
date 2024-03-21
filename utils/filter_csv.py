import pandas as pd

df = pd.read_csv('../ensemble/datasets/fastmri_multicoil_knee_train.csv')
dffilt = df.loc[df['enc_y'] >= 368]
dffilt.to_csv('../ensemble/datasets/fastmri_multicoil_knee_train_filtered.csv')