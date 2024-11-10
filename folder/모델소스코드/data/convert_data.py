import argparse
import os
import shutil
import json
from process_sql import get_sql

TABLE_FOLDER = "원천데이터"
LABEL_FOLDER = "라벨링데이터"


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema, table):
        self._schema = schema
        self._table = table
        self._idMap = self._map(self._schema, self._table)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema, table):
        column_names_original = table['column_names_original']
        table_names_original = table['table_names_original']
        #print 'column_names_original: ', column_names_original
        #print 'table_names_original: ', table_names_original
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                idMap = {'*': i}
            else:
                key = table_names_original[tab_id].lower()
                val = col.lower()
                idMap[key + "." + val] = i

        for i, tab in enumerate(table_names_original):
            key = tab.lower()
            idMap[key] = i

        return idMap


def get_schemas_from_json(fpath):
    # with open(fpath) as f:
    #     data = json.load(f)
    with open(fpath) as f:
        # data = json.load(f)["data"]
        data = json.load(f)
    db_names = [db['db_id'] for db in data]

    tables = {}
    schemas = {}
    for db in data:
        db_id = db['db_id']
        schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
        column_names_original = db['column_names_original']
        table_names_original = db['table_names_original']
        tables[db_id] = {'column_names_original': column_names_original,
                         'table_names_original': table_names_original}
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[table] = cols
        schemas[db_id] = schema

    return schemas, db_names, tables


def get_path(root_dir, target_dir):
    label_path = os.path.join(root_dir, target_dir)
    for src_folder in os.listdir(label_path):
        src_path = os.path.join(label_path, src_folder)
        for extract_type in os.listdir(src_path):
            parent_path = os.path.join(label_path, src_folder, extract_type)
            for file_name in os.listdir(parent_path):
                yield parent_path, file_name


def load_json_file(json_file_path, _type="list"):
    if os.path.exists(json_file_path):
        with open(json_file_path) as rfile:
            json_data = json.load(rfile)
    else:
        if _type == "list":
            json_data = []
        else:
            json_data = []

    return json_data


def table_check(data):
    column_origin_dict = dict()
    column_dict = dict()
    table_dict = dict()

    if type(data["table_names_original"]) == str:
        data["table_names_original"] = [data["table_names_original"]]
    if type(data["table_names"]) == str:
        data["table_names"] = [data["table_names"]]

    column_origin_names = data["column_names_original"]
    column_names = data["column_names"]
    column_types = data["column_types"]
    col_names = []

    for i, col in enumerate(column_names):
        table_index = col[0]
        col_name = col[1]
        col_origin_name = column_origin_names[i][1]
        col_names.append(col_name)

        if col_name not in column_dict:
            column_dict[col_name] = 1
        else:
            column_dict[col_name] += 1

        if col_origin_name not in column_origin_dict:
            column_origin_dict[col_origin_name] = 1
        else:
            column_origin_dict[col_origin_name] += 1

        if table_index not in table_dict:
            table_dict[table_index] = 1
        else:
            table_dict[table_index] += 1

    if len(column_types) != len(column_origin_names):
        return False

    col_names = list(set(col_names))
    col_names.sort()
    col_count = len(column_dict)

    if col_count > 50:
        return False

    return True


def get_labeled_data(schemas, tables, db_id_list, data):
    db_id = data["db_id"]

    if db_id in db_id_list:
        try:
            spider_format = {
                "db_id": str(),
                "utterance_id": str(),
                "hardness": str(),
                "utterance_type": str(),
                "query": str(),
                "query_toks": list(),
                "query_toks_no_value": list(),
                "question": str(),
                "question_toks": list(),
                "values": list(),
                "cols": list(),
                "sql": dict()
            }
            sql = data["query"]
            schema = schemas[db_id]
            table = tables[db_id]
            schema = Schema(schema, table)
            sql_label = get_sql(schema, sql)
            spider_format["db_id"] = db_id
            spider_format["utterance_id"] = data["utterance_id"]
            spider_format["hardness"] = data["hardness"]
            spider_format["utterance_type"] = data["utterance_type"]
            spider_format["query"] = sql
            spider_format["question"] = data["utterance"]
            spider_format["question_toks"] = data["utterance"].split()
            spider_format["sql"] = sql_label

            # evaluate_hardness, _ = evaluator.eval_hardness(sql_label)
            gold_data = sql + "\t" + db_id + "\n"

            return spider_format, gold_data

        except Exception as e:
            return False, False

    else:
        return False, False


def main(config):
    labeling_json_path = os.path.join(config.data_path, config.name + ".json")
    annotation_json_path = os.path.join(config.data_path, "tables.json")
    annotation_list = load_json_file(annotation_json_path)
    print("# of", annotation_json_path, "-", len(annotation_list))
    labeling_list = []
    labeled_list = []
    db_id_list = [annotation_data["db_id"] for annotation_data in annotation_list]
    # for folder in os.listdir(config.src_folder):
    root_path = os.path.join(config.src_folder)
    # copy database
    os.makedirs(os.path.join(config.data_path, config.database_path), exist_ok=True)

    """
        set labeling_list, annotation_list
    """
    for parent_path, file_name in get_path(root_path, LABEL_FOLDER):
        src_path = os.path.join(parent_path, file_name)
        # print(src_path)
        target_dict = load_json_file(src_path, _type="dict")
        labeling_list.extend(target_dict["data"])
        # labeling_list.extend(target_dict["data"])
        # if file_name == config.annotation_file_name:
        #     for annotation_data in target_dict["data"]:
        #         db_id = annotation_data["db_id"]
        #         if db_id not in db_id_list and table_check(annotation_data):
        #             append_count += 1
        #             annotation_list.append(annotation_data)
        #             db_id_list.append(db_id)
        # elif file_name == config.labeling_file_name:
        #     labeling_list.extend(target_dict["data"])

    """
        copy sqlite folder 
    """
    append_count = 0
    exist_count = 0
    for parent_path, target_db in get_path(root_path, TABLE_FOLDER):
        src_path = os.path.join(parent_path, target_db)

        dst_path = os.path.join(os.path.join(config.data_path, config.database_path), target_db)

        if src_path.endswith(".json"):
            # print(src_path, dst_path)
            target_dict = load_json_file(src_path, _type="dict")
            for annotation_data in target_dict["data"]:
                db_id = annotation_data["db_id"]
                if db_id not in db_id_list and table_check(annotation_data):
                    append_count += 1
                    annotation_list.append(annotation_data)
                    db_id_list.append(db_id)
                else:
                    exist_count += 1
        else:
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)

    print("# of",  config.name, "tables -", append_count)

    """
        save annotation json file "tables.json" 
    """
    with open(annotation_json_path, 'w') as wf:
        json.dump(annotation_list, wf, indent=4, ensure_ascii=False)
    schemas, db_names, tables = get_schemas_from_json(annotation_json_path)

    """
        get sql_label data
    """
    gold_data_list = []
    except_count = 0
    for data in labeling_list:
        labeled_data, gold_data = get_labeled_data(schemas, tables, db_id_list, data)
        if labeled_data:
            labeled_list.append(labeled_data)
            gold_data_list.append(gold_data)
        else:
            except_count += 1

    """
        save labeling json file "train.json", "train_gold.sql"
    """
    with open(labeling_json_path, 'w') as wf:
        json.dump(labeled_list, wf, indent=4, ensure_ascii=False)

    with open(labeling_json_path.split(".json")[0] + "_gold.sql", 'wt') as out:
        out.writelines(gold_data_list)

    print("# of",  config.name, "except count -", except_count)
    # # save data_dict
    # annotation_json_path = os.path.join(config.data_path, config.annotation_file_name)
    # labeling_json_path = os.path.join(config.data_path, config.labeling_file_name)
    # with open(annotation_json_path, 'w') as wf:
    #     json.dump(annotation_dict, wf, indent=4, ensure_ascii=False)
    # with open(labeling_json_path, 'w') as wf:
    #     json.dump(labeling_dict, wf, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", default="download/1.Training", type=str, help="data_folder")
    parser.add_argument("--name", choices=["train", "valid", "dev"], type=str, help="set name")
    parser.add_argument("--data_path", default="nia", type=str, help="data folder name")
    parser.add_argument("--database_path", default="database", type=str, help="database folder")
    parser.add_argument("--annotation_file_name", default="nl2sql_data_annotation.json", type=str, help="annotation file name")
    parser.add_argument("--labeling_file_name", default="nl2sql_data_labeling.json", type=str, help="labeling file name")
    args = parser.parse_args()
    main(args)
