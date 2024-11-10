import os
import re
import json
import numpy as np
import pandas as pd
import pprint
import pickle
from collections import OrderedDict

from tqdm import tqdm
from eval_final.evaluation import Evaluator

from nltk.stem import PorterStemmer
porter = PorterStemmer()

import utils_ko

EX_LIST = ["ex_groupby", "ex_orderby", "ex_limit", "ex_where", "ex_having", "ex_iuen"]

# define constant/list/mappdings
AGG = ["none", "max", "min", "count", "sum", "avg"]
AGG_NUM = len(AGG)
ARI = ["none", "-", "+", "*", "/"]
ARI_NUM = len(ARI)
OPS = ["none", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exits"]
OPS_NUM = len(OPS)

# if score is not good, to erase parallel
SPC = ["none", "union", "intersect", "except", "where", "having", "parallel"]
SPC_NUM = len(SPC)
SPC_ID_MAP = dict()
for i, spc in enumerate(SPC) :
    SPC_ID_MAP[spc] = i

# 1_cycle 1%
# TABLE_PATH = "data/nia/table/tables.json"
# # TABLE_PATH = "data/nia/table/220816_all_tables.json"
# COLUMN_DICT = "data/nia/table/220818_all_columns.json"
# TRAIN_PATH = "data/nia/1_cycle/train.json"
# VALID_PATH = "data/nia/1_cycle/valid.json"
# DEV_PATH = "data/nia/1_cycle/dev.json"
# EXCEED_PATH = "data/nia/error_case/220818_exceed_list.json"
# EXCEED_OVERLAP = True
# TABLE_CONCAT = True

# # 1_cycle 10%
# TABLE_PATH = "data/nia/table/220921_all_tables.json"
# COLUMN_DICT = "data/nia/table/220921_all_columns.json"
# TRAIN_PATH = "data/nia/10_cycle/train.json"
# VALID_PATH = "data/nia/10_cycle/valid.json"
# DEV_PATH = "data/nia/10_cycle/dev.json"
# EXCEED_PATH = "data/nia/error_case/220921_exceed_list.json"
#
# EXCEED_OVERLAP = True


ORDER_MAP = {
    "asc" : 0
    , "desc" : 1
}
ORDER_MAP_REV = {
    0: "asc",
    1: "desc"
}
CONJ_MAP = {
    "and" : 0
    , "or" : 1
}
CONJ_MAP_REV = {
    0: "and",
    1: "or"
}

# MAX_NUM = {
#     "table_id": 26,         # max num of tables in Schema
#     "table_num": 5,         # max num of 'JOIN'
#     "column_num": 26,       # max num of tables in Schema
#     "select": 6,            # max num of columns in 'SELECT'
#     "groupby": 4,           # max num of columns in 'GROUP By'
#     "orderby": 3,           # max num of columns in 'ORDER BY'
#     "where": 4,             # max num of columns in 'WHERE'
#     "having": 2,            # max num of columns in 'HAVING'
#     "spc_id": 4             # max num of depth (sub-sql, INTERSECT, UNION, EXCEPT)
# }

COL_SIZE = 3
ignore_idx = -100

NUM_MAP = {
      "one" : 1
    , "two" : 2
    , "three" : 3
    , "four" : 4
    , "five" : 5
    , "six" : 6
    , "seven" : 7
    , "eight" : 8
    , "nine" : 9
    , "ten" : 10
    , "eleven" : 11
    , "twelve" : 12
}

STR_TO_NUM = {
      1 : "one"
    , 2 : "two"
    , 3 : "three" 
    , 4 : "four"
    , 5 : "five" 
    , 6 : "six"
    , 7 : "seven" 
    , 8 : "eight" 
    , 9 : "nine" 
    , 10 : "ten"
    , 11 : "eleven"
    , 12 : "twelve"
}

IEUN_MAP_REV = {
    0: "intersect",
    1: "except",
    2: "union",
    3: ""
}

IEUN_MAP = {
    "intersect": 0,
    "except": 1,
    "union": 2,
    "none": 3
}


def flat_list(l) :
    return [y for x in l for y in x]


def update_max_val(k, v, max_num) :
    if k not in max_num :
        max_num[k] = 0
    max_num[k] = max(max_num[k], v)


def update_count_val(k, count) :
    if k not in count :
        count[k] = 0
    count[k] += 1


def check_table_id(n, table) :
    "Handle table_id with non-numeric type, i.e. string"
    clean_n = re.sub("[0-9]", "", str(n))
    if clean_n=="" :
        return n
    clean_n = re.sub("[^0-9a-zA-Z]", "", n).lower()
    names = table["table_names_original"]
    for i, name in enumerate(names) :
        if clean_n==re.sub("[^0-9a-zA-Z]", "", name).lower() :
            return i
    return 0


def check_column_id(n, table) :
    "Handle column_id with non-numeric type, i.e. string"
    clean_n = re.sub("[0-9]", "", str(n))
    if clean_n=="" :
        return n
    clean_n = re.sub("[^0-9a-zA-Z]", "", n).lower()
    names = table["column_names_original"]
    names[0] = "all"
    for i, name in enumerate(names) :
        if clean_n==re.sub("[^0-9a-zA-Z]", "", name).lower() :
            return i
    return 0


def get_dummy_label(size, dummy_val=-100) :
    label = np.zeros(size, dtype=np.int16) 
    label.fill(dummy_val)
    return label


def get_col_unit_label(col_unit, table) :
    if col_unit is None :
        ignore_label = np.zeros(COL_SIZE, dtype=np.int8)
        ignore_label.fill(ignore_idx)
        return ignore_label
    else :
        agg_id, col_id, dist = col_unit
        col_id = check_column_id(col_id, table)
        return np.array([
            int(agg_id), int(col_id), int(dist)
        ], dtype=np.int16)


def get_val_unit_label(val_unit, table) :
    unit_op, col_unit1, col_unit2 = val_unit
    return int(unit_op) \
        , get_col_unit_label(col_unit1, table) \
        , get_col_unit_label(col_unit2, table)


def get_condition_label(tokenizer, question_toks, clause, max_num, count, table, clause_key="where") :
    labels = tuple()
    cond_num = ignore_idx

    # cond_op_labels = get_dummy_label((MAX_NUM[clause_key], 2))
    # unit_labels = get_dummy_label(MAX_NUM[clause_key])
    # con1_labels = get_dummy_label((MAX_NUM[clause_key], COL_SIZE))
    # con2_labels = get_dummy_label((MAX_NUM[clause_key], COL_SIZE))
    # conj_label = get_dummy_label(MAX_NUM[clause_key])
    # val_label = get_dummy_label((MAX_NUM[clause_key], 2))
    # val_span_label = get_dummy_label((MAX_NUM[clause_key], 2, 2))
    cond_op_labels = get_dummy_label((max_num[clause_key], 2))
    unit_labels = get_dummy_label(max_num[clause_key])
    con1_labels = get_dummy_label((max_num[clause_key], COL_SIZE))
    con2_labels = get_dummy_label((max_num[clause_key], COL_SIZE))
    conj_label = get_dummy_label(max_num[clause_key])
    val_label = get_dummy_label((max_num[clause_key], 2))
    val_span_label = get_dummy_label((max_num[clause_key], 2, 2))

    sqls = [None, None]
    unique_conjs = set()
    if len(clause) > 0 :
        cond_num = int((len(clause) + 1)/2) - 1
        update_count_val(clause_key+"_"+str(cond_num), count)
        idx = 0
        for i, cond in enumerate(clause) :
            if i % 2 != 0 :
                conj_label[idx] = CONJ_MAP[cond.lower()]
                unique_conjs.add(cond.lower())
                update_count_val(cond.lower(), count)
                idx += 1    # only increase after and/or keyword
            else :
                not_op, op_id, val_unit, val1, val2 = cond
                # check first val is sql or not
                if isinstance(val1, dict) :
                    val_label[idx, 0] = 1
                    sqls[0] = val1
                elif val1 is not None :
                    val_label[idx, 0] = 0
                    val_span_label[idx][0][0] = 0
                    val_span_label[idx][0][1] = 0
                else :
                    val_label[idx, 0] = 0
                # check second val is sql or not
                if isinstance(val2, dict) :
                    val_label[idx, 1] = 1
                    sqls[1] = val2
                elif val2 is not None :
                    val_label[idx, 1] = 0
                    val_span_label[idx][1][0] = 0
                    val_span_label[idx][1][1] = 0
                else :
                    val_label[idx, 1] = 0
                cond_op_labels[idx][0] = int(not_op)
                cond_op_labels[idx][1] = int(op_id)
                unit_op, con_unit1, con_unit2 = get_val_unit_label(val_unit, table)
                update_count_val("{}_unit_{}".format(clause_key, unit_op), count)
                unit_labels[idx] = unit_op
                con1_labels[idx] = con_unit1
                con2_labels[idx] = con_unit2
            update_max_val(clause_key, cond_num, max_num)
    update_count_val("{}_{}".format(clause_key, ' '.join(list(unique_conjs))), count)
    labels += (cond_num, )
    labels += (cond_op_labels, )
    labels += (unit_labels, )
    labels += (con1_labels, )
    labels += (con2_labels, )
    labels += (conj_label, )
    labels += (val_label, )
    labels += (val_span_label, )

    """
        where values (in BETWEEN) have sub-sql both
    """
    a = 0
    for _s in sqls:
        if _s is not None:
            a += 1
    if a > 1:
        print(a, sqls)
        exit(-1)

    return labels, sqls


def parse_sql(sql, max_num, table, question_toks, tokenizer, all_labels, sql_labels, count, spc=["none"]) :
    ################################
    # Assumptions:
    #   1. sql is correct
    #   2. only table name has alias
    #   3. only one intersect/union/except
    #
    # val: number(float)/string(str)/sql(dict)
    # col_unit: (agg_id, col_id, isDistinct(bool))
    # val_unit: (unit_op, col_unit1, col_unit2)
    # table_unit: (table_type, col_unit/sql)
    # cond_unit: (not_op, op_id, val_unit, val1, val2)
    # condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
    # sql {
    #   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
    #   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
    #   'where': condition
    #   'groupBy': [col_unit1, col_unit2, ...]
    #   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
    #   'having': condition
    #   'limit': None/limit value
    #   'intersect': None/sql
    #   'except': None/sql
    #   'union': None/sql
    # }
    labels = tuple()
    #############################################################
    # make task labels for clause existance prediction
    labels += (int(len(sql["groupBy"])>0), )         # bg
    labels += (int(len(sql["orderBy"])>0), )         # bo
    labels += (int(sql["limit"] is not None), )      # bl
    labels += (int(len(sql["where"])>0), )           # bw
    labels += (int(len(sql["having"])>0), )          # bh

    iuen = None
    if int(sql["intersect"] is not None) :
        labels += (IEUN_MAP["intersect"], )    # intersect
        iuen = "intersect"
    elif int(sql["union"] is not None) :
        labels += (IEUN_MAP["union"], )    # union
        iuen = "union"
    elif int(sql["except"] is not None) :
        labels += (IEUN_MAP["except"], )    # except
        iuen = "except"
    else :
        labels += (IEUN_MAP["none"], )    # none

    if iuen is not None :
        parse_sql(sql[iuen], max_num, table, question_toks, tokenizer, all_labels, sql_labels, count, spc+[iuen])
    # 6
    #############################################################
    # make table clause labels
    from_clause = sql["from"]
    from_tables = from_clause["table_units"]
    table_ids = []
    for t in from_tables :
        table_type, n = t
        # table_type = "table_unit" | "sql"
        # we are going to ignore "sql" type and extract only tables from it
        if table_type!="table_unit" :
            update_count_val("from_clause_non_table_unit", count)
            for _, n in t[1]["from"]["table_units"] :
                table_ids.append(check_table_id(n, table))
        else :
            table_ids.append(check_table_id(n, table))
    table_ids = list(set(table_ids))
    table_num = len(table_ids) - 1
    update_count_val("table_num_{}".format(table_num), count)
    # max_num["table_num"] = max(max_num["table_num"], len(table_ids))
    # table_id_label = get_dummy_label(MAX_NUM["table_id"], dummy_val=0)
    if max_num["table_id"] == 2:
        table_id_label = get_dummy_label(max_num["table_id"] + 1, dummy_val=0)
    else:
        table_id_label = get_dummy_label(max_num["table_id"], dummy_val=0)

    for i in table_ids :
        table_id_label[i] = 1
    # 9
    #############################################################
    # make select clause labels
    select_clause = sql["select"]
    select_dist_label = (int(select_clause[0]), )
    select_num_label = (len(select_clause[1]) - 1, )
    update_count_val("sel_col_num"+str(select_num_label), count)
    update_max_val("sel_cond", labels[-1], max_num)
    update_count_val("sel_cond_num_"+str(labels[-1]), count)
    # sel_agg_label = get_dummy_label(MAX_NUM["select"])
    # sel_unit_label = get_dummy_label(MAX_NUM["select"])
    # sel_con1_label = get_dummy_label((MAX_NUM["select"], COL_SIZE))
    # sel_con2_label = get_dummy_label((MAX_NUM["select"], COL_SIZE))
    sel_agg_label = get_dummy_label(max_num["select"])
    sel_unit_label = get_dummy_label(max_num["select"])
    sel_con1_label = get_dummy_label((max_num["select"], COL_SIZE))
    sel_con2_label = get_dummy_label((max_num["select"], COL_SIZE))
    for i, (agg_id, val_unit) in enumerate(select_clause[1]) :
        sel_agg_label[i] = agg_id
        update_count_val("sel_cond_agg_{}".format(agg_id), count)
        unit_op, con_unit1, con_unit2 = get_val_unit_label(val_unit, table)
        sel_unit_label[i] = unit_op
        update_count_val("sel_op_{}".format(unit_op), count)
        sel_con1_label[i] = con_unit1
        sel_con2_label[i] = con_unit2

    # max_num["select"] = max(max_num["select"], len(select_clause[1]))
    # 15
    #############################################################
    # make order-by clause labels
    orderby_clause = sql["orderBy"]
    ord_sort = ignore_idx
    ord_cond_num = ignore_idx
    # ord_unit_labels = get_dummy_label(MAX_NUM["orderby"])
    # ord_col1_labels = get_dummy_label((MAX_NUM["orderby"], COL_SIZE))
    # ord_col2_labels = get_dummy_label((MAX_NUM["orderby"], COL_SIZE))
    ord_unit_labels = get_dummy_label(max_num["orderby"])
    ord_col1_labels = get_dummy_label((max_num["orderby"], COL_SIZE))
    ord_col2_labels = get_dummy_label((max_num["orderby"], COL_SIZE))
    if len(orderby_clause) > 0 :
        order, conds = orderby_clause
        ord_cond_num = len(conds) - 1
        ord_sort = ORDER_MAP[order]
        for i, val_unit in enumerate(conds) :
            unit_op, con_unit1, con_unit2 = get_val_unit_label(val_unit, table)
            ord_unit_labels[i] = unit_op
            update_count_val("ord_op_{}".format(unit_op), count)
            ord_col1_labels[i] = con_unit1
            ord_col2_labels[i] = con_unit2
        # max_num["orderby"] = max(max_num["orderby"], len(conds))
    # update_max_val("ord_cond", ord_cond_num, max_num)
    #############################################################
    # make group-by clause labels
    groupby_clause = sql["groupBy"]
    #grb_num = len(groupby_clause)
    grb_num = ignore_idx
    # grb_col_labels = get_dummy_label((MAX_NUM["groupby"], COL_SIZE))
    grb_col_labels = get_dummy_label((max_num["groupby"], COL_SIZE))
    if len(groupby_clause) > 0 :
        grb_num = len(groupby_clause) - 1
        for i, col_unit in enumerate(groupby_clause) :
            grb_col_labels[i] = get_col_unit_label(col_unit, table)
    # max_num["groupby"] = max(max_num["groupby"], grb_num)
    #############################################################
    # make limit clause labels
    limit_clause = sql["limit"]
    is_top1 = ignore_idx
    val_pos = ignore_idx
    if limit_clause is not None :
        is_top1 = int(limit_clause==1)
        if not is_top1 :
            val_pos = 0
            for pos, val in enumerate(question_toks) :
                #val = tokenizer._convert_id_to_token(tok)
                #val = val.lower().replace("##", "")
                val = val.replace("##", "")
                if str(val)==str(limit_clause) :
                    val_pos = pos
                if val in NUM_MAP and str(NUM_MAP[val])==str(limit_clause):
                    val_pos = pos
    #############################################################
    # make where clause labels
    # labels += (cond_num, )
    # labels += (cond_op_labels, )
    # labels += (unit_labels, )
    # labels += (col1_labels, )
    # labels += (col2_labels, )
    # labels += (ao_label, )
    where_clause = sql["where"]
    where_label, sqls = get_condition_label(tokenizer, question_toks, where_clause, max_num, count, table, "where")

    for sql_ in sqls :
        if sql_ is not None :
            parse_sql(sql_, max_num, table, question_toks, tokenizer, all_labels, sql_labels, count, spc+["where"])
            # break

    #############################################################
    # make having clause labels
    having_clause = sql["having"]
    having_label, sqls = get_condition_label(tokenizer, question_toks, having_clause, max_num, count, table, "having")
    for sql_ in sqls :
        if sql_ is not None :
            parse_sql(sql_, max_num, table, question_toks, tokenizer, all_labels, sql_labels, count, spc+["having"])
            # break

    #############################################################
    spc_seq = [[SPC_ID_MAP[s] for s in spc]]
    # spc_seq, spc_mask = utils_ko.pad_sequence(
    #         spc_seq, max_seq=MAX_NUM["spc_id"], pad_max=True
    #     )
    spc_seq, spc_mask = utils_ko.pad_sequence(
            spc_seq, max_seq=max_num["spc_id"], pad_max=True
        )

    # merge all labels
    labels += (table_num, ) # table number label
    labels += (table_id_label, ) # table id labels
    labels += select_dist_label # select global disticnt label
    labels += select_num_label # select condition number label
    labels += (sel_agg_label, ) 
    labels += (sel_unit_label, )
    labels += (sel_con1_label, )
    labels += (sel_con2_label, )
    labels += (ord_sort, )
    labels += (ord_cond_num, )
    labels += (ord_unit_labels, )
    labels += (ord_col1_labels, )
    labels += (ord_col2_labels, )
    labels += (grb_num, grb_col_labels, )
    labels += (is_top1, val_pos, )
    labels += where_label
    labels += having_label
    """
        spc_seq, spc_mask, bg, bo, bl, bw, bh, ieun, table, select, order, group, limit, where, having
    """
    labels = (spc_seq[0], spc_mask[0], ) + labels

    all_labels.append(labels)
    sql_labels.append(sql)


def _get_tok_ids(tokenizer, text) :
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, ids


def get_table_pkl(config, tokenizer, data):
    def make_concat_name(_table, concat=True) :
        column = _table["column_names"]
        _table = _table["table_names"]
        names = [[] for _ in range(len(_table))]
        toks = [[] for _ in range(len(_table))]
        toks_origin = [[] for _ in range(len(_table))]
        for col in column :
            t, name = col
            if t==-1 :
                continue
            if str(_table[t]) not in name and concat:
                name = str(_table[t]) + ' ' + name
            names[t].append(name)
            tok, ids = _get_tok_ids(tokenizer, name)
            toks[t].append(ids)
            toks_origin[t].append(tok)
        return names, toks, toks_origin

    # # 나중에 parse_sql을 샘플별로 실행하면서 각 max가 설정됨
    # max_num = {
    #     "table_id": 0,      # max num of tables in Schema
    #     "table_num": 0,     # max num of 'JOIN'
    #     "column_num": 0,    # max num of total columns in Schema
    #     "select": 0,        # max num of columns in 'SELECT'
    #     "groupby": 0,       # max num of columns in 'GROUP By'
    #     "orderby": 0,       # max num of columns in 'ORDER BY'
    #     "where": 0,         # max num of columns in 'WHERE'
    #     "having": 0,        # max num of columns in 'HAVING'
    #     "spc_id": 0         # max num of depth (sub-sql, INTERSECT, UNION, EXCEPT)
    # }

    pkl_path = os.path.join(config.pkl_path, config.pkl_name)

    if not os.path.exists(config.pkl_path):
        try:
            os.makedirs(config.pkl_path, exist_ok=True)
        except OSError:
            print("\nError making pkl directory\n")
            exit(-1)

    if not os.path.exists(pkl_path) :
        tables = dict()

        print("making pkl -", pkl_path)
        for table in tqdm(data) :

            # if table["db_id"]=="formula_1" :
            #     table["table_names_original"] = ['races', 'drivers', 'status', 'seasons', 'constructors', 'constructorStandings', 'results', 'driverStandings', 'constructorResults', 'qualifying', 'circuits', 'pitStops', 'lapTimes']
            #     table["column_names"] = table["column_names_original"]

            col_names, col_toks, col_toks_origin = make_concat_name(table, concat=config.table_concat)

            org_column_names = [
                table["table_names_original"][t_id]+'.'+name
                for t_id, name in table["column_names_original"]
            ]

            """
                db_id (스키마 id) = {
                    # 스키마를 구성하는 테이블들의 전체 column id
                    column_table_map = [ -1 (*), 0, 0, 0, 1, 1, 2, 2, 2, 2, ... n, n ]
                }
            """
            tables[table["db_id"]] = {
                "column_table_map" : [l[0] for l in table["column_names_original"]]
                , "column_name" : col_names
                , "column_name_toks" : col_toks
                , "column_name_toks_origin" : col_toks_origin
                , "foreign_key" : table["foreign_keys"]
                , "table_names_original" : table["table_names_original"]
                , "column_names_original" : org_column_names
            }

            # update_max_val("column_num", len(table["column_names"]), max_num)
            # update_max_val("table_num", len(table["table_names_original"]), max_num)

            table_id_map = {}
            for i, name in enumerate(table["table_names_original"]) :
                table_id_map[name] = i
            column_id_map = {}
            for i, (table_id, name) in enumerate(table["column_names_original"]) :
                table_name = table["table_names_original"][table_id].lower()
                name = name.lower()
                column_id_map["__{}.{}__".format(table_name, name)] = i
            tables[table["db_id"]]["table_id_map"] = table_id_map
            tables[table["db_id"]]["column_id_map"] = column_id_map

        with open(pkl_path, "wb") as handle :
            pickle.dump(tables, handle)

    else :
        print("loading {} table pkl...".format(pkl_path))
        with open(pkl_path, "rb") as handle :
            tables = pickle.load(handle)

        for _, table_dict in tables.items():
            column_num = 0
            for _col_names in table_dict["column_name"]:
                column_num += len(_col_names)

            # update_max_val("column_num", column_num, max_num)
            # update_max_val("table_num", len(table_dict["column_name"]), max_num)

    return tables
    #
    # return tables, max_num


def _load_samples(tokenizer, filename, tables, table_names, max_num, count):
    print("\nloading {} dataset...".format(filename))
    # print("tokenizer -", tokenizer)
    target = os.path.basename(filename).split(".json")[0]
    exceed_col_dict = dict()
    samples = []
    token_num_exceed_samples = 0
    token_num_exceed_db = set()
    sample_with_sub_sql = 0
    evaluator = Evaluator()

    exceed_list = []
    a = 0
    c = 0

    with open(os.path.join(filename), "r") as handle:
        data = json.load(handle)
    # aa = 0
    for utt_id, sample in enumerate(data):
        # process each sample
        # aa += 1
        # process question
        utt_id = sample["utterance_id"] if "utterance_id" in sample else "utt_" + str(utt_id + 1)

        question = ' '.join(sample["question_toks"])
        question_toks, question_ids = _get_tok_ids(tokenizer, question)
        tokens = [tokenizer.cls_id] + question_ids
        origin_tokens = ['[CLS]'] + question_toks
        q_len = len(question_ids)

        table_names.add(sample["db_id"])
        # dev_samples.append(
        #     {
        #         "query": sample["query"],
        #         "sql": sample["sql"]
        #     }
        # )
        target_table = tables[sample["db_id"]]

        column_num = 0
        for _col_names in target_table["column_name"]:
            column_num += len(_col_names)

        # update_max_val("column_num", column_num, max_num)

        c_len = []
        # [CLS] Q [SEP] * [SEP] C_1 [SEP] C_2 .... C_N [SEP]
        # tokens = tokens + [tokenizer.sep_id] + [tokenizer._convert_token_to_id('*')]
        tokens = tokens + [tokenizer.sep_id] + [tokenizer.convert_tokens_to_ids('*')]
        origin_tokens = origin_tokens + ['[SEP]'] + ['*']
        c_len.append([1])
        schema_token_len = 0
        # for column in target_table["column_name_toks"]:
        #     _c_len = []
        #     for toks in column:
        #         tokens = tokens + [tokenizer.sep_id] + toks
        #         _c_len.append(len(toks))
        #     c_len.append(_c_len)
        #     b += len(_c_len)
        for column, column_origin in zip(target_table["column_name_toks"], target_table["column_name_toks_origin"]):
            _c_len = []
            for _toks, _toks_origin in zip(column, column_origin):
                tokens = tokens + [tokenizer.sep_id] + _toks
                origin_tokens = origin_tokens + ['[SEP]'] + _toks_origin
                _c_len.append(len(_toks))
            c_len.append(_c_len)
            schema_token_len += len(_c_len)

        # print(sample["db_id"])
        # print(c_len)
        # print(tokens)
        # print(origin_tokens)
        # print()
        # if aa == 100:
        #     exit(-1)
        # table length
        t_len = len(c_len)

        sql = sample["sql"]
        all_labels = []
        sql_labels = []

        """
            sample의 sql마다 재귀sql 개수가 다르기 때문에 all_labels는 서로 다른 크기를 가짐
            [ spc, bg, bo, bl, bw, bh, ng, no, ns, nu, nh, IUEN ]
            [
                [ 
                  [ NONE, SELECT A FROM B WHERE 1 ]
                  [ NONE + WHERE, SELECT max(C) ]
                ]
            ]
        """
        parse_sql(sql, max_num, target_table, question_toks, tokenizer, all_labels, sql_labels, count)
        # if len(sql_labels) > 1:
        #     # print(sample["utterance_id"])
        #     for ll, aa in zip(sql_labels, all_labels):
        #         print(ll)
        #         print(aa[0])
        #         print()
        #     exit(-1)

        update_max_val("tokens", len(tokens), max_num)
        if len(tokens) > 512:

            token_num_exceed_db.add(sample["db_id"])
            token_num_exceed_samples += 1

            # if column_dict:
            #     exceed_col_len = column_dict[sample["db_id"]]
            # else:
            #     exceed_col_len = 0
            #     for column_names in target_table["column_name"]:
            #         exceed_col_len += len(column_names)

            # if column_num not in exceed_col_dict:
            #     exceed_col_dict[column_num] = 1
            # else:
            #     exceed_col_dict[column_num] += 1
            # print(c_len)
            aaa = 0
            for _c in c_len[1:]:
                # for _t in _c:
                aaa += len(_c)
            # print(sample["db_id"], len(origin_tokens))
            # print(origin_tokens)
            # print()
            # print()
            exceed_template = OrderedDict(
                {
                    "data": target,
                    "utterance_id": utt_id,
                    "utterance": sample["question"],
                    "db_id": sample["db_id"],
                    "column_names": target_table["column_name"],
                    "token_length": len(tokens),
                    "token": " ".join(origin_tokens)
                }
            )
            exceed_list.append(exceed_template)
            continue

        if len(all_labels) > 1:
            sample_with_sub_sql += 1

        # for label in all_labels:
        #     spc = label[0]
        #     update_max_val("spc_depth", len(spc), max_num)
        update_max_val("tot_sql_num", len(all_labels), max_num)

        # 샘플마다 query 토큰 수, 참조하는 table 수, 참조하는 column 수가 다름
        samples.append({
            "tokens": tokens
            , "q_len": q_len
            , "t_len": t_len
            , "c_len": c_len
            , "label": all_labels
            , "table_map": target_table["column_table_map"]
            , "db_id": sample["db_id"]
            , "utt_id": utt_id
            , "hardness": sample["hardness"] if "hardness" in sample else evaluator.eval_hardness(sql)
            , "query": sample["query"]
        })
        # a = max(a, len(tokens))
        # c = max(schema_token_len, c)

    print("total samples", len(samples))
    print("token_num_exceed_samples", token_num_exceed_samples)
    # print(token_num_exceed_db)
    # print("number of samples with sub-sql ", sample_with_sub_sql)
    #
    # print("\n")
    # exceed_col_dict = sorted(exceed_col_dict.items(), key=lambda item: item[0], reverse=False)
    # for k, v in exceed_col_dict:
    #     print(k, v)
    #
    # print("\n")

    return samples, exceed_list


def _analyze_sample(sql, max_num, table, spc_depth):
    def _get_condition_sub_sql(clause, clause_key):
        sub_sql = [None, None]
        if len(clause) > 0:
            cond_num = 0
            for _i, cond in enumerate(clause):
                if _i % 2 == 0:
                    cond_num += 1
                    not_op, op_id, val_unit, val1, val2 = cond
                    # check first val is sql or not
                    if isinstance(val1, dict):
                        sub_sql[0] = val1
                    # check second val is sql or not
                    if isinstance(val2, dict):
                        sub_sql[1] = val2

            update_max_val(clause_key, cond_num, max_num)

        """
            where values (in BETWEEN) have sub-sql both
        """
        a = 0
        for _s in sub_sql:
            if _s is not None:
                a += 1
        if a > 1:
            print(a, sub_sql)
            exit(-1)

        return sub_sql

    iuen = None
    if int(sql["intersect"] is not None) :
        iuen = "intersect"
    elif int(sql["union"] is not None) :
        iuen = "union"
    elif int(sql["except"] is not None) :
        iuen = "except"

    if iuen is not None :
        spc_depth = _analyze_sample(sql[iuen], max_num, table, spc_depth)

    # update num of 'JOIN'
    table_ids = []
    for t in sql["from"]["table_units"]:
        table_type, n = t
        if table_type != "table_unit":
            for _, n in t[1]["from"]["table_units"]:
                table_ids.append(check_table_id(n, table))
        else:
            table_ids.append(check_table_id(n, table))
    table_ids = list(set(table_ids))
    update_max_val("table_num", len(table_ids), max_num)

    # update num of max SELECT column
    update_max_val("select", len(sql["select"][1]), max_num)

    # update num of max ORDER BY column
    if len(sql["orderBy"]) > 0 :
        update_max_val("orderby", len(sql["orderBy"][1]), max_num)

    # update num of max GROUP BY column
    if len(sql["groupBy"]) > 0 :
        update_max_val("groupby", len(sql["groupBy"]), max_num)

    for sql_ in _get_condition_sub_sql(sql["where"], "where"):
        if sql_ is not None:
            _analyze_sample(sql_, max_num, table, spc_depth)
            # break

    for sql_ in _get_condition_sub_sql(sql["having"], "having"):
        if sql_ is not None:
            _analyze_sample(sql_, max_num, table, spc_depth)
            # break

    return spc_depth + 1


def _read_samples(_path, max_num, tables):
    """
        TODO: if you want to more optimize, use tokenizer and erase sample which has over sized tokens
    """
    with open(os.path.join(_path), "r") as handle:
        data = json.load(handle)

    for utt_id, sample in enumerate(data):
        target_table = tables[sample["db_id"]]

        column_num = 0
        for _col_names in target_table["column_name"]:
            column_num += len(_col_names)
        update_max_val("column_num", column_num, max_num)
        update_max_val("table_id", len(target_table["column_name"]), max_num)

        spc_depth = 1
        sql = sample["sql"]
        spc_depth = _analyze_sample(sql, max_num, target_table, spc_depth)
        update_max_val("spc_id", spc_depth, max_num)

        # if len(all_labels) > 1:
        #     sample_with_sub_sql += 1
        #
        # for label in all_labels:
        #     spc = label[0]
        #     update_max_val("spc_depth", len(spc), max_num)
        # update_max_val("tot_sql_num", len(all_labels), max_num)


def get_max_num(config, tables):
    """
        set max_num and return

        if: use_fixed_max_num option -> max_num in config file
        else: fit max_num in dataset
    """
    if config.use_fixed_max_num:
        print("\nuse fixed maximum number stats...")
        print(config.max_num, '\n')
        return config.max_num

    max_num = {
        "table_id": 0,      # max num of tables in Schema
        "table_num": 0,     # max num of 'JOIN'
        "column_num": 0,    # max num of total columns in Schema
        "select": 0,        # max num of columns in 'SELECT'
        "groupby": 0,       # max num of columns in 'GROUP By'
        "orderby": 0,       # max num of columns in 'ORDER BY'
        "where": 0,         # max num of columns in 'WHERE'
        "having": 0,        # max num of columns in 'HAVING'
        "spc_id": 0         # max num of depth (sub-sql, INTERSECT, UNION, EXCEPT)
    }

    if type(config.train_path) is str:
        train_paths = [config.train_path]
    else:
        train_paths = config.train_path

    for train_path in train_paths:
        _read_samples(train_path, max_num, tables)

    if config.valid_path:
        _read_samples(config.valid_path, max_num, tables)

    _read_samples(config.dev_path, max_num, tables)

    config.max_num = max_num

    if config.train:
        print("\nfit maximum number stats...")
        print(max_num, '\n')

    return max_num


def load_spider(config, tokenizer) :
    #global SPC_ID
    # print("SPC codes")
    # print(SPC_ID_MAP)
    # print("")
    train_tables = set()
    valid_tables = set()
    dev_tables = set()
    # dev_samples = list()
    with open(config.table_path, "r") as handle :
        data = json.load(handle)

    # column_dict = dict()
    # if config.column_dict:
    #     with open(config.column_dict, 'r') as handle:
    #         column_dict = json.load(handle)

    count = {}

    tables = get_table_pkl(config, tokenizer, data)
    total_exceed_list = []
    dataset = {}

    max_num = get_max_num(config, tables)

    """
        set train
    """
    if type(config.train_path) is list:
        dataset["train"] = []
        for train_path in config.train_path:
            samples, exceed_list = _load_samples(tokenizer, train_path, tables, train_tables, max_num, count)
            dataset["train"].extend(samples)
            total_exceed_list.extend(exceed_list)
    else:
        samples, exceed_list = _load_samples(tokenizer, config.train_path, tables, train_tables, max_num, count)
        dataset["train"] = samples
        total_exceed_list.extend(exceed_list)

    """
        set valid
    """
    if config.valid_path:
        samples, exceed_list = _load_samples(tokenizer, config.valid_path, tables, valid_tables, max_num, count)
        total_exceed_list.extend(exceed_list)
    else:
        samples, _ = _load_samples(tokenizer, config.dev_path, tables, valid_tables, max_num, count)
    dataset["valid"] = samples

    """
        set dev
    """
    samples, exceed_list = _load_samples(tokenizer, config.dev_path, tables, dev_tables, max_num, count)
    dataset["dev"] = samples
    total_exceed_list.extend(exceed_list)

    # dataset["train"] = _load_samples("train_extracted.json", tables, train_tables, max_num, count)
    # dataset["dev"] = _load_samples("dev_extracted.json", tables, dev_tables, max_num, count)

    # print("\ntrain table num", len(train_tables))
    # print("valid table num", len(valid_tables))
    # print("dev table num", len(dev_tables))
    # print("intersect num", len(set(train_tables).intersection(set(dev_tables))))

    # if not os.path.isfile(config.exceed_path):
    #     with open(config.exceed_path, "w") as f:
    #         json.dump(total_exceed_list, f, ensure_ascii=False, indent=4)
    #
    # if config.exceed_overlap:
    #     with open(config.exceed_path, "w") as f:
    #         json.dump(total_exceed_list, f, ensure_ascii=False, indent=4)

    # print()
    # print(max_num)
    # print()
    # exit(-1)

    # print("maximum number stats...")
    # print(pprint.pprint(max_num))
    # print("count things...")
    # print(pprint.pprint(count))
    # exit(-1)

    # return dataset, tables, train_tables, dev_tables
    return dataset, tables, train_tables, dev_tables
