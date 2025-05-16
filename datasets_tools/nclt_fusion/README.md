## Data Preprocessing

1. We have provided preprocessed dataset in the `./data/nclt_fusion`(consistent with the [paper](#anchor3)).
2. If you want to re-download the data and process it, use the following steps

```
# download sensor data and ground truth
python ./datasets_tools/nclt_fusion/down.py --sen
python ./datasets_tools/nclt_fusion/down.py --gt
mkdir -p ./data/NCLT/sensor_data
ls ./data/NCLT/sensor_data_tar/*.tar.gz | xargs -I {} tar xzvf {} -C ./data/NCLT/sensor_data
python ./datasets_tools/nclt_fusion/preprocess.py
```
