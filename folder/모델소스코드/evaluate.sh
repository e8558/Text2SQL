#!/usr/bin/env bash

save_ckpt="save_model/ckpt_221129_fitting_decay_09"

###
# parameters
match="match"
table="data/nia/tables.json"
db="data/nia/database/"
file="data/nia/dev.json"
batch_size=32
device=0
###

. parse_options.sh || echo "Can't find parse_options.sh" | exit 1

gold=${save_ckpt}"/dev_gold_parsed.sql"
pred=${save_ckpt}"/predictions.txt"
info=${save_ckpt}"/dev_gold_parsed_info.txt"

python main_ko.py --save_ckpt $save_ckpt --batch_size $batch_size --device $device
python eval_final/evaluation.py --gold $gold --pred $pred --db $db --table $table --etype $match
python eval_final/dump_csv.py --gold $gold --pred $pred --db $db --table $table --file $file --info $info