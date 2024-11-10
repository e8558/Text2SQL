import argparse
import json
import pandas as pd
import os
from evaluation import Evaluator, build_foreign_key_map_from_json, rebuild_sql_col, rebuild_sql_val, build_valid_col_units
from process_sql import get_schema, Schema, get_sql
from copy import deepcopy


def dump_csv(gold, predict, db_dir, file, info, kmaps):
    with open(gold) as f:
        glist = [l.strip().split("\t") for l in f.readlines() if len(l.strip()) > 0]

    with open(predict) as f:
        plist = [l.strip().split("\t") for l in f.readlines() if len(l.strip()) > 0]

    with open(info) as f:
        utt_list = [l.strip() for l in f.readlines() if len(l.strip()) > 0]

    with open(file) as f:
        sample_dict = {sample["utterance_id"]: sample for sample in json.load(f)}

    csv_path = os.path.join(os.path.dirname(predict), "predictions.csv")

    csv_data_dict = {
        "utt_id": [],
        "gold_sql": [],
        "pred_sql": [],
        "exact_match_without_value": [],
        "exact_match_with_value": [],
        "hardness": [],
    }

    partial_hardness = [
        "comp1",
        "comp2",
        "others"
    ]
    for ph_ in partial_hardness:
        csv_data_dict[ph_] = []

    partial_types = [
        "select",
        "select(no AGG)",
        "where",
        "where(no OP)",
        "group(no Having)",
        "group",
        "order",
        "and/or",
        "IUEN",
        "keywords",
    ]
    for type_ in partial_types:
        csv_data_dict[type_] = []


    evaluator = Evaluator()
    count_em_with_value = 0
    count_em_without_value = 0
    for i, (p, g, utt_id) in enumerate(zip(plist, glist, utt_list)):
        p_str = p[0]
        g_str, db = g
        db_name = db
        db = os.path.join(db_dir, db, db + ".sqlite")
        schema = Schema(get_schema(db))
        g_sql = get_sql(schema, g_str)
        hardness = evaluator.eval_hardness(g_sql)
        details = evaluator.eval_hardness_details(g_sql)
        sample_hardness = sample_dict[utt_id]["hardness"]
        sample_hardness = "extra" if sample_hardness.startswith("extra") else sample_hardness

        try:
            p_sql = get_sql(schema, p_str)
        except:
            # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
            p_sql = {
                "except": None,
                "from": {"conds": [], "table_units": []},
                "groupBy": [],
                "having": [],
                "intersect": None,
                "limit": None,
                "orderBy": [],
                "select": [False, []],
                "union": None,
                "where": [],
            }

        exact_score_with_value = evaluator.eval_exact_match(deepcopy(p_sql), deepcopy(g_sql))

        kmap = kmaps[db_name]
        g_valid_col_units = build_valid_col_units(g_sql["from"]["table_units"], schema)
        g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
        p_valid_col_units = build_valid_col_units(p_sql["from"]["table_units"], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)

        exact_score_without_value = evaluator.eval_exact_match(p_sql, g_sql)
        partial_scores = evaluator.partial_scores

        if exact_score_with_value:
            count_em_with_value += 1
        if exact_score_without_value:
            count_em_without_value += 1

        for type_ in partial_types:
            csv_data_dict[type_].append(partial_scores[type_]["f1"])

        csv_data_dict["utt_id"].append(utt_id)
        csv_data_dict["gold_sql"].append(g_str)
        csv_data_dict["pred_sql"].append(p_str)
        csv_data_dict["exact_match_without_value"].append(1 if exact_score_without_value else 0)
        csv_data_dict["exact_match_with_value"].append(1 if exact_score_with_value else 0)
        csv_data_dict["hardness"].append(sample_hardness)

        for ph_, d in zip(partial_hardness, details):
            csv_data_dict[ph_].append(d)

    # print(count_em_without_value)
    # print(count_em_with_value)
    df = pd.DataFrame(csv_data_dict)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", dest="file", type=str)
    parser.add_argument("--info", dest="info", type=str)
    parser.add_argument("--gold", dest="gold", type=str)
    parser.add_argument("--pred", dest="pred", type=str)
    parser.add_argument("--table", dest="table", type=str)
    parser.add_argument("--db", dest="db", type=str)
    args = parser.parse_args()

    gold = args.gold
    pred = args.pred
    db_dir = args.db
    file = args.file
    info = args.info
    table = args.table

    kmaps = build_foreign_key_map_from_json(table)

    dump_csv(gold, pred, db_dir, file, info, kmaps)


if __name__ == '__main__':
    main()
