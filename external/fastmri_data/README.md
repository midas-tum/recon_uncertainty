# Install requirements
```
pip install -r requirements.txt
```

# Environment variables
```
export FASTMRI_ROOT=<your-fastmri-root-dir>
```

# fastMRI setup
- `$FASTMRI_ROOT`
  - knee
    - data
  - brain
    - data

# Generate csv file
```
python ./data_processing/create_dataset_csv.py --data-path ${FASTMRI_ROOT} --csv-file=../datasets/singlecoil_knee_train.csv --dataset knee/data/singlecoil_train  
```

```
python ./data_processing/create_dataset_csv.py --data-path ${FASTMRI_ROOT} --csv-file=../datasets/singlecoil_knee_val.csv --dataset knee/data/singlecoil_val  
```

```
python ./data_processing/create_dataset_csv.py --data-path ${FASTMRI_ROOT} --csv-file=../datasets/singlecoil_knee_test.csv --dataset knee/data/singlecoil_test  
```

# Test dataloader
First, adapt the `<fastmri-root-dir>` in `fastmri_dataloader/config.yml`

```
python -m unittest fastmri_dataloader/fastmri_dataloader_tf.py
python -m unittest fastmri_dataloader/fastmri_dataloader_th.py
```