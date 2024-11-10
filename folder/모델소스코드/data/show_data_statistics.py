import os
import json
from collections import OrderedDict


FILE_PATH = [
    "nia/train.json",
    "nia/valid.json",
    "nia/dev.json"
]


def update_map(_dict, _id, _h):
    if _id not in _dict:
        _dict[_id][_h] = 1
    else:
        _dict[_id][_h] += 1


def round_avg(a, b, r=2):
    return round((a / b) * 100, r)


def show_statistics(_dict, file_path, hardness_index_dict):
    basename = os.path.basename(file_path).split(".json")[0]

    print(basename)
    header_str = ""
    total_count_dict = {}
    total_count_rev = {}
    for i, h in enumerate(hardness_index_dict):
        total_count_rev[i] = h
        total_count_dict[h] = 0

    for utt_type, count_list in _dict.items():
        for i, c in enumerate(count_list):
            total_count_dict[total_count_rev[i]] += c

    total_str = "total"
    total_count = total_count_dict["total"]
    for i, h in enumerate(hardness_index_dict):
        header_str += "\t" + str(h).rjust(11) + "    "
        total_str += "\t" + str(total_count_dict[h]).rjust(7)
        total_hardness_count = round_avg(total_count_dict[h], total_count)
        total_str += " (" + str(total_hardness_count) + ")"
        # if i + 1 < len(hardness_index_dict):
        #     total_hardness_count = round_avg(total_count_dict[h], total_count)
        #     total_str += " (" + str(total_hardness_count) + ")"

    print(header_str)
    for utt_type, count_list in _dict.items():
        row_str = utt_type
        total_utt_count = count_list[-1]
        for i, c in enumerate(count_list):
            if hardness_index_dict["total"] == i:
                avg = round_avg(c, total_count)
            else:
                avg = round_avg(c, total_utt_count)
            row_str += "\t" + str(c).rjust(7) + " (" + str(avg) + ")"
        print(row_str)
    print(total_str)
    print("\n")


def main(file_path_list):
    for file_path in file_path_list:
        sql_path = os.path.join(file_path)
        with open(sql_path) as inf:
            sql_data = json.load(inf)

        data_dict = {}

        hardness_dict = OrderedDict({
            "easy": [],
            "medium": [],
            "hard": [],
            "extra": []
        })

        hardness_index_dict = {h: i for i, h in enumerate(hardness_dict)}
        hardness_index_dict["total"] = len(hardness_dict)

        utt_type_dict = OrderedDict({
            "Wht": [0 for _ in hardness_dict] + [0],
            "Whn": [0 for _ in hardness_dict] + [0],
            "Whr": [0 for _ in hardness_dict] + [0],
            "Hch": [0 for _ in hardness_dict] + [0],
            "Who": [0 for _ in hardness_dict] + [0]
        })

        for data in sql_data:
            utt_id = data["utterance_id"]
            utt_type = utt_id.split("_")[0]
            hardness = "extra" if data["hardness"].startswith("extra") else data["hardness"]

            if utt_id not in data_dict:
                data_dict[utt_id] = data
                hardness_dict[hardness].append(utt_id)
            else:
                print("\nError] duplicated utt_id\n")
                exit(-1)

            update_map(utt_type_dict, utt_type, hardness_index_dict["total"])

        for hardness in hardness_dict:
            for utt_id in hardness_dict[hardness]:
                # utt_id = data_dict[utt_id]["utterance_id"]
                utt_type = utt_id.split("_")[0]
                update_map(utt_type_dict, utt_type, hardness_index_dict[hardness])

        show_statistics(utt_type_dict, file_path, hardness_index_dict)

if __name__ == '__main__':
    main(FILE_PATH)

