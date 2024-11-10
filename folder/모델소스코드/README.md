# RYANSQL
## Overview
### Model description

This model is developed by referring to RYANSQL, a text-to-SQL system for complex, cross-domain databases.

Reference Paper: Choi et al., [RYANSQL: Recursively Applying Sketch-based Slot Fillings for Complex Text-to-SQL in Cross-Domain Databases](https://arxiv.org/abs/2004.03125), 2020

The system is submitted to [SPIDER leaderboard](https://yale-lily.github.io/spider). The system and its minor improved version RYANSQL v2 is ranked at second and fourth place (as of February 2020).

The system does NOT use any database records, which make it more acceptable to the real world company applications.

### Model architecture

![캡처](img/ryansql.png)

### input

- inputs are expected to be ELECTRA encoded query and database (schema)
- Shape: `(B, 512, 768)`

### output

- outputs are expected to be ELECTRA
- Shape: `(B, 512, 768)`

## Set up Environment

    pip install -f https://download.pytorch.org/whl/torch_stable.html -r requirements.txt

## Preprocessing

Download KoNL2SQL Data in 'data/{FOLDER}'

    mkdir data/download

Convert KoNL2SQL Data for using model

    cd data
    python convert_data.py --src_folder download/01.Training/ --name train
    python convert_data.py --src_folder download/02.Validation/ --name valid
    python convert_data.py --src_folder download/03.Test/ --name dev

If you want to know statistics of Datasets, Use the script below
    
    python show_data_statistics.py 


## Training

If you want to change parameters, Edit the config file below 
    
    "ko": {
        "pretrained_model": "monologg/koelectra-base-v3-discriminator",
        "pkl_path": "pkl/koelectra-base-v3-discriminator",
        "pkl_name": "data.pkl",
        "table_path": "data/nia/tables.json",
        "train_path": "data/nia/train.json",
        "valid_path": "data/nia/valid.json",
        "dev_path": "data/nia/dev.json",
        "patience": 5,
        "learning_rate": 1e-6,
        "table_concat": false,
        "use_fixed_max_num": false,  
        "max_num": {
            "table_id" : 2,
            "table_num" : 2,
            "column_num" : 48,
            "select" : 6,
            "groupby" : 2,
            "orderby" : 2,
            "where" : 4,
            "having" : 1,
            "spc_id" : 2
        }
      },

Train with RYANSQL model

    python main_ko.py --train --save_ckpt {SAVE_CKPT} --batch_size {BATCH_SIZE} --device {DEVICE}
    ex) python main_ko.py --train --save_ckpt save_model/best_ckpt --batch_size 16 --device 0

## Inference


Inference with RYANSQL model using evaluation script from [Spider](https://github.com/taoyds/spider)

    ./evaluate.py --save_ckpt {SAVE_CKPT} --batch_size {BATCH_SIZE} --device {DEVICE}
    ex) ./evluate.py --save_ckpt save_model/best_ckpt --batch_size 16 --device 0

You can get csv file below

    {SAVE_CKPT}/best_ckpt/predictions.csv

## License
`RYANSQL`은 `Apache-2.0` 라이선스 하에 공개되어 있습니다. 모델 및 코드를 사용할 경우 라이선스 내용을 준수해주세요. 라이선스 전문은 LICENSE 파일에서 확인하실 수 있습니다.
