import numpy as np
import data_ko as Data
from collections import OrderedDict

DECODE_ERROR_IGNORE = True
DECODE_ERROR_STR = "#DECODE_ERROR#"

VALUE_STR = "'#VALUE#'"
WHERE_NESTED_STR = "'#WHERE_NESTED#'"
HAVING_NESTED_STR = "'#HAVING_NESTED#'"

PRED_NUM_ORIGIN = False
PRED_WHERE_ARI = True


class IncrementIndex :
    def __init__(self, max_num) :
        self.i = -1
        self.max_num = max_num
    
    def get(self) :
        self.i += 1
        if self.i >= self.max_num :
            raise Exception("[IncrementIndex] Exceed maximum index number!")
        return self.i


def flat_list(l) :
    return [y for x in  l for y in x]


def make_mask(x, pad_idx, decode=False):
    "Create a mask to hide padding and future words."
    mask = (x!=pad_idx)
    if decode :
        size = x.shape[-1]
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        mask = np.expand_dims(mask, axis=1) & (subsequent_mask == 0)
    return mask.astype('uint8')


def pad_sequence(x, max_seq=64, pad_idx=0, get_mask=True, decode=False, pad_max=False) :
    """
    padding given sequence with maximum length 
    generate padded sequence and mask
    """ 
    seq_len = np.array([min(len(seq), max_seq) for seq in x])
    if not pad_max :
        max_seq = max(seq_len)
    pad_seq = np.zeros((len(x), max_seq), dtype=np.int64)
    pad_seq.fill(pad_idx)
    for i, seq in enumerate(x):
        pad_seq[i, :seq_len[i]] = seq[:seq_len[i]]
    if get_mask :
        mask = make_mask(pad_seq, pad_idx, decode)
    else :
        mask = None
    return pad_seq, mask


def check_val_pred(batch_idx
                    , num
                    , preds
                    , unit_acc
                    , agg1_acc
                    , col1_acc
                    , dist1_acc
                    , table_map
                    , used_table
                   , units
                    ) :
    # 0 unit
    # 1 agg1
    # 2 col1
    # 3 dist1
    # 4 agg2
    # 5 col2
    # 6 dist2
    unit_pred, unit_label = preds[0]

    pred_set = set()
    gold_set = set()
    for i in range(num) :
        ord_unit = [
            0,  # op_unit
            [],  # agg, col, dist
            [],  # agg, col, dist
        ]

        ord_unit_1 = ord_unit[1]
        ord_unit_2 = ord_unit[2]

        pred_cond = []
        gold_cond = []

        # op_unit
        pred = np.argmax(unit_pred[batch_idx], axis=-1)[i]
        pred_cond.append(pred)
        gold_cond.append(unit_label[batch_idx][i])
        ord_unit[0] = pred

        # ord_unit_2
        if pred != 0 :
            for comp in preds[4:] :
                p, l = comp
                p = np.argmax(p[batch_idx][i], axis=-1)
                pred_cond.append(p)
                gold_cond.append(l[batch_idx][i])
                ord_unit_2.append(p)

        unit_acc.append(pred==unit_label[batch_idx][i])

        def get_val_result(batch_idx, num_idx, preds, acc, pred_cond, gold_cond) :
            for i, pred in enumerate(preds) :
                p, l = pred
                p = np.argmax(p[batch_idx][num_idx], axis=-1)
                pred_cond.append(p)
                gold_cond.append(l[batch_idx][num_idx][0])
                acc[i].append(pred_cond[-1]==gold_cond[-1])
                ord_unit_1.append(p)

        # ord_unit_1
        get_val_result(
            batch_idx
            , i
            , preds[1:4]
            , [agg1_acc, col1_acc, dist1_acc]
            , pred_cond, gold_cond)

        # for comp, acc in zip(preds[1:4], [agg1_acc, col1_acc, dist1_acc]):
        #     p, l = comp
        #     p = np.argmax(p[batch_idx][i], axis=-1)
        #     pred_cond.append(p)
        #     gold_cond.append(l[batch_idx][i])
        #     acc.append(pred_cond[-1]==gold_cond[-1])
        #     ord_unit_1.append(p)

        pred_set.add('-'.join([str(v) for v in pred_cond]))
        gold_set.add('-'.join([str(v) for v in gold_cond]))
        units.append(ord_unit)

    return pred_set, gold_set


def check_pred(batch_size, preds, table_map, tables, utt_ids) :
    results = dict()
    sqls = [
        {
            "select": [],
            "orderBy": [],
            "groupBy": [],
            "limit": [],
            "where": {
                "cond_units": [],
                "and_or": []
            },
            "having": {
                "cond_units": [],
                "and_or": []
            },
            "table": [],
            "iuen": int()           # 0: intersect / 1: union / 2: except / 3: none
        } for _ in range(batch_size)
    ]
    pred_vals = dict()
    label_vals = dict()
    used_tables = [set() for _ in range(batch_size)]
    # correctness of sample predictions
    sample_pred = np.ones(batch_size)==1
    ##############################################################

    # measure clause exist prediction accuracy
    for name in Data.EX_LIST :
        pred, label = preds[name]
        pred = np.argmax(pred, axis=-1)
        pred_vals[name] = pred
        label_vals[name] = label
        results[name] = pred==label
        # having clause only exists when groupby clause exists
        if name=="ex_having" :
            results[name] = results[name]|(results["ex_groupby"]&pred_vals["ex_groupby"]==0)

        if name =="ex_iuen":
            for batch_index, pred_iuen in enumerate(pred):
                sqls[batch_index]["iuen"] = pred_iuen
        sample_pred *= results[name]

    ##############################################################
    #"""
    # measure select clause accuracy
    if "select" in preds :
        pred = preds["select"]

        sel_dist, sel_num = pred[:2]
        pred_sel_dist = np.argmax(sel_dist[0], axis=-1)
        pred_sel_num = np.argmax(sel_num[0], axis=-1)
        results["sel_dist"] = pred_sel_dist==sel_dist[1]
        results["sel_num"] = pred_sel_num==sel_num[1]

        for i, is_dist in enumerate(pred_sel_dist):
            sqls[i]["select"].append(True if is_dist else False)
            sqls[i]["select"].append([])

        agg, ari = pred[2:4]
        select_acc = []
        sel_ari_acc = []
        sel_agg_acc = []
        
        sel_agg1_acc = []
        sel_col1_acc = []
        sel_dist1_acc = []

        for i in range(batch_size) :
            pred_set = set()
            gold_set = set()
            sql_select = sqls[i]["select"][1]

            if PRED_NUM_ORIGIN:
                range_pred_sel_num = sel_num[i] + 1
            else:
                range_pred_sel_num = pred_sel_num[i]+1

            for j in range(range_pred_sel_num) :
            # for j in range(sel_num[i]+1) :
                pred_cond = []
                gold_cond = []

                """
                    [unit_op, con_unit_1, con_unit_2]
                """
                sel_unit = [
                    0,          # sel_agg
                    [
                        0,      # unit_op
                        [],     # agg, col, dist
                        [],     # agg, col, dist
                    ]
                ]
                cond_unit_1 = sel_unit[1][1]
                cond_unit_2 = sel_unit[1][2]

                pred_agg = np.argmax(agg[0][i][j], axis=-1)
                pred_cond.append(pred_agg)
                gold_cond.append(agg[1][i][j])
                sel_agg_acc.append(pred_cond[-1]==gold_cond[-1])

                sel_unit[0] = pred_agg

                p, l = pred[4]
                p = np.argmax(p[i][j], axis=-1)
                pred_cond.append(p)
                gold_cond.append(l[i][j][0])
                sel_agg1_acc.append(pred_cond[-1]==gold_cond[-1])
                cond_unit_1.append(p)

                p, l = pred[5]
                p = np.argmax(p[i][j], axis=-1)
                used_tables[i].add(table_map[i][p])
                pred_cond.append(p)
                gold_cond.append(l[i][j][0])
                sel_col1_acc.append(pred_cond[-1]==gold_cond[-1])
                cond_unit_1.append(p)

                p, l = pred[6]
                p = np.argmax(p[i][j], axis=-1)
                pred_cond.append(p)
                gold_cond.append(l[i][j][0])
                sel_dist1_acc.append(pred_cond[-1]==gold_cond[-1])
                cond_unit_1.append(p)
                cond_unit_1[-1] = True if cond_unit_1[-1] else False

                p_ari = np.argmax(ari[0], axis=-1)[i][j]
                pred_cond.append(p_ari)

                if p_ari != 0 :
                    for comp in pred[7:] :
                        p, l = comp
                        p = np.argmax(p[i][j], axis=-1)
                        pred_cond.append(p)
                        cond_unit_2.append(p)
                    cond_unit_2[-1] = True if cond_unit_2[-1] else False

                gold_cond.append(ari[1][i][j])
                if ari[1][i][j]!=0 :
                    for comp in pred[7:] :
                        p, l = comp
                        gold_cond.append(l[i][j])

                sel_ari_acc.append(p_ari==ari[1][i][j])
                sel_unit[1][0] = p_ari

                pred_set.add('-'.join([str(v) for v in pred_cond]))
                gold_set.add('-'.join([str(v) for v in gold_cond]))
                sql_select.append(sel_unit)

            if len(pred_set)==len(gold_set) and len(pred_set-gold_set)==0 :
                select_acc.append(True)
            else :
                select_acc.append(False)

        results["select_clause"] = np.array(select_acc)
        results["sel_agg_acc"] = np.array(sel_agg_acc)
        results["sel_ari_acc"] = np.array(sel_ari_acc)
        results["sel_agg1_acc"] = np.array(sel_agg1_acc)
        results["sel_col1_acc"] = np.array(sel_col1_acc)
        results["sel_dist1_acc"] = np.array(sel_dist1_acc)
        sample_pred *= results["select_clause"]
        sample_pred *= results["sel_dist"]
        sample_pred *= results["sel_num"]

    ##############################################################
    # orderby clause accuracy
    #"""
    if "orderby" in preds :
        # pred = preds["orderBy"] = [
        #     None,
        #     [
        #
        #     ]
        # ]
        pred = preds["orderby"]
        sort_pred, sort_label = pred[0]
        num_pred, num_label = pred[1]
        val_preds = pred[2:]

        # clause_acc = []
        order_clause_acc = []
        sort_acc = []
        num_acc = []
        unit_acc = []
        agg1_acc = []
        col1_acc = []
        dist1_acc = []
        val_acc = []

        for i in range(batch_size):
            if label_vals["ex_orderby"][i]==0 :
                # clause_acc.append(results["ex_orderby"][i])
                order_clause_acc.append(results["ex_orderby"][i])
                continue

            # asc / desc
            pred = np.argmax(sort_pred[i], axis=-1)
            sort_acc.append(pred==sort_label[i])
            sqls[i]["orderBy"] = [
                pred,   # asc / desc
                []      # col_units
            ]

            # ord clause num
            pred = np.argmax(num_pred[i], axis=-1)
            num_acc.append(pred==num_label[i])

            # ord units
            pred_set, gold_set = check_val_pred(i, pred+1, val_preds
                                                , unit_acc
                                                , agg1_acc
                                                , col1_acc
                                                , dist1_acc
                                                , table_map[i]
                                                , used_tables
                                                , sqls[i]["orderBy"][1]
                                                )

            if len(pred_set)==len(gold_set) and len(pred_set-gold_set)==0 :
                val_acc.append(True)
            else :
                val_acc.append(False)

            # clause_acc.append(
            #     sort_acc[-1] and num_acc[-1] and val_acc[-1]
            # )
            order_clause_acc.append(
                sort_acc[-1] and num_acc[-1] and val_acc[-1]
            )

            # print(utt_ids[i])
            # print(sqls[i]["orderby"])
            # print()

        # results["ord_clause"] = np.array(clause_acc)
        results["ord_clause"] = np.array(order_clause_acc)
        results["ord_sort"] = np.array(sort_acc)
        results["ord_num"] = np.array(num_acc)
        results["ord_unit"] = np.array(unit_acc)
        results["ord_agg1"] = np.array(agg1_acc)
        results["ord_col1"] = np.array(col1_acc)
        results["ord_dist1"] = np.array(dist1_acc)
        results["ord_val_unit"] = np.array(val_acc)
        # sample_pred *= np.array(clause_acc)
        sample_pred *= results["ord_clause"]

    ##############################################################
    # groupby clause accuracy
    #"""
    if "groupby" in preds :
        pred = preds["groupby"]
        num_pred, num_label = pred[0]
        col_pred, col_label = pred[1]

        # clause_acc = []
        group_clause_acc = []
        num_acc = []
        col_acc = []

        for i in range(batch_size):
            if label_vals["ex_groupby"][i]==0 :
                # clause_acc.append(results["ex_groupby"][i])
                group_clause_acc.append(results["ex_groupby"][i])
                continue

            pred = np.argmax(num_pred[i], axis=-1)
            num_acc.append(pred==num_label[i])

            pred_set = set()
            gold_set = set()
            for j in range(pred+1) :
                # print('label -', col_label[i][j])
                # print('pred -', col_pred[i][j])
                grb_col_pred = np.argmax(col_pred[i][j], axis=-1)
                # print(grb_col_pred)
                # print()
                pred_set.add(grb_col_pred)
                gold_set.add(col_label[i][j])
                sqls[i]["groupBy"].append(
                    [
                        0,
                        grb_col_pred,
                        False
                    ]
                )

            if len(pred_set)==len(gold_set) and len(pred_set-gold_set)==0 :
                col_acc.append(True)
            else :
                col_acc.append(False)

            # clause_acc.append(
            #     num_acc[-1] and col_acc[-1]
            # )
            group_clause_acc.append(
                num_acc[-1] and col_acc[-1]
            )

        # results["grb_clause"] = np.array(clause_acc)
        results["grb_clause"] = np.array(group_clause_acc)
        results["grb_num"] = np.array(num_acc)
        results["grb_col"] = np.array(col_acc)
        sample_pred *= results["grb_clause"]

    ##############################################################
    # limit clause accuracy
    if "limit" in preds :
        pred = preds["limit"]
        top1, pos = pred
        top1_pred, top1_label = top1
        pos_pred, pos_label = pos
        
        # clause_acc = []
        lim_clause_acc = []
        lim_top1_acc = []
        lim_pos_acc = []
        for i in range(batch_size):
            if label_vals["ex_limit"][i]==0 :
                # clause_acc.append(results["ex_limit"][i])
                lim_clause_acc.append(results["ex_limit"][i])
                continue
            # top1
            pred = np.argmax(top1_pred[i], axis=-1)
            lim_top1_acc.append(pred==top1_label[i])
            sqls[i]["limit"] = [pred]

            # topN (N != 1)
            if top1_label[i]==0 :
                pred = np.argmax(pos_pred[i], axis=-1)
                lim_pos_acc.append(pred==pos_label[i])
                sqls[i]["limit"].append(pred)
            else :
                lim_pos_acc.append(True)
                sqls[i]["limit"].append(0)

            # clause_acc.append(
            #     lim_top1_acc[-1] and lim_pos_acc[-1]
            # )
            lim_clause_acc.append(
                lim_top1_acc[-1] and lim_pos_acc[-1]
            )
        
        results["lim_top1"] = np.array(lim_top1_acc)
        results["lim_pos"] = np.array(lim_pos_acc)
        # results["lim_clause"] = np.array(clause_acc)
        results["lim_clause"] = np.array(lim_clause_acc)
        sample_pred *= results["lim_clause"]
    ##############################################################
    # where clause accuracy
    #"""
    def eval_clause(clause_type, preds, results, table_map, used_tables, sqls, where_clause=False) :
        pred = preds[clause_type]
        num_pred, num_label = pred[0]

        clause_acc = []
        num_acc = []

        conj_acc = []
        not_cond_acc = []
        nest_1_acc = []
        nest_2_acc = []
        ari_acc = []

        agg1_acc = []
        col1_acc = []
        dist1_acc = []

        cond_acc = []

        conj, not_cond, cond, nest_1, nest_2, ari = pred[1:7]

        conj_pred, conj_label = conj
        for i in range(batch_size) :
            if where_clause:
                pred_and_or_list = sqls[i]["where"]["and_or"]
                pred_cond_units = sqls[i]["where"]["cond_units"]
            else:
                pred_and_or_list = sqls[i]["having"]["and_or"]
                pred_cond_units = sqls[i]["having"]["cond_units"]

            if label_vals["ex_"+clause_type][i]==0 :
                clause_acc.append(results["ex_"+clause_type][i])
                continue

            condition_num_pred = np.argmax(num_pred[i], axis=-1)
            num_acc.append(condition_num_pred==num_label[i])

            if PRED_NUM_ORIGIN:
                conj_pred_batch = np.argmax(conj_pred[i][:num_label[i]], axis=-1)
            else:
                conj_pred_batch = np.argmax(conj_pred[i][:condition_num_pred], axis=-1)

            conj_pred_batch = conj_pred_batch.tolist()
            conj_label_batch = conj_label[i][:num_label[i]].tolist()
            conj_acc.append(
                len(conj_pred_batch)==len(conj_label_batch) and \
                    len(set(conj_pred_batch)-set(conj_label_batch))==0 and \
                    len(set(conj_label_batch)-set(conj_pred_batch))==0
            )

            for conj_pred_index in conj_pred_batch:
                pred_and_or_list.append(Data.CONJ_MAP_REV[conj_pred_index])

            pred_set = set()
            gold_set = set()

            if PRED_NUM_ORIGIN:
                range_pred_cond_num = num_label[i]+1
            else:
                range_pred_cond_num = condition_num_pred+1

            for j in range(range_pred_cond_num) :
                pred_cond = []
                gold_cond = []

                """
                    cond_unit = [
                        true/false,         # NOT
                        where_op,
                        [
                            unit_op,
                            [
                                agg,
                                column,
                                isDist
                            ],
                            [
                                agg,
                                column,
                                isDist
                            ]                            
                        ],
                        val_1,
                        val_2
                    ]
                """
                where_having_unit = [
                    False,      # not operation
                    0,          # condition operation ["none", "between", "=", ">", "<", ">=", "<=", "!=", "in" ... ]
                    [
                        0,      # unit_op
                        [],     # agg, col, dist
                        [],     # agg, col, dist
                    ],
                    False,      # val_1 is_nest
                    False       # val_2 is_nest
                ]
                cond_unit_1 = where_having_unit[2][1]
                cond_unit_2 = where_having_unit[2][2]

                pred_not_op = np.argmax(not_cond[0][i][j], axis=-1)
                pred_cond.append(pred_not_op)
                gold_cond.append(not_cond[1][i][j][0])
                not_cond_acc.append(pred_cond[-1]==gold_cond[-1])
                where_having_unit[0] = True if pred_not_op else False

                pred_op = np.argmax(cond[0][i][j], axis=-1)
                pred_cond.append(pred_op)
                gold_cond.append(cond[1][i][j][0])
                cond_acc.append(pred_cond[-1]==gold_cond[-1])
                where_having_unit[1] = pred_op

                is_nest_1 = np.argmax(nest_1[0][i][j], axis=-1)
                pred_cond.append(is_nest_1)
                gold_cond.append(nest_1[1][i][j][0])
                nest_1_acc.append(pred_cond[-1]==gold_cond[-1])
                where_having_unit[3] = True if is_nest_1 else False

                is_nest_2 = np.argmax(nest_2[0][i][j], axis=-1)
                pred_cond.append(is_nest_2)
                gold_cond.append(nest_2[1][i][j][0])
                nest_2_acc.append(pred_cond[-1]==gold_cond[-1])
                where_having_unit[4] = True if is_nest_2 else False

                pred_ari = np.argmax(ari[0][i][j], axis=-1)
                label_ari = ari[1][i][j]
                pred_cond.append(pred_ari)
                gold_cond.append(label_ari)
                ari_acc.append(pred_cond[-1]==gold_cond[-1])
                where_having_unit[2][0] = pred_ari

                p, l = pred[7]
                p = np.argmax(p[i][j], axis=-1)
                pred_cond.append(p)
                gold_cond.append(l[i][j][0])
                agg1_acc.append(pred_cond[-1]==gold_cond[-1])
                cond_unit_1.append(p)

                p, l = pred[8]
                p = np.argmax(p[i][j], axis=-1)
                pred_cond.append(p)
                used_tables[i].add(table_map[i][p])
                gold_cond.append(l[i][j][0])
                col1_acc.append(pred_cond[-1]==gold_cond[-1])
                cond_unit_1.append(p)

                p, l = pred[9]
                p = np.argmax(p[i][j], axis=-1)
                pred_cond.append(p)
                gold_cond.append(l[i][j][0])
                dist1_acc.append(pred_cond[-1]==gold_cond[-1])
                cond_unit_1.append(p)
                cond_unit_1[-1] = True if cond_unit_1[-1] else False

                if pred_ari:
                    for _comp in pred[10:]:
                        p, _ = _comp
                        p = np.argmax(p[i][j], axis=-1)
                        cond_unit_2.append(p)
                        if PRED_WHERE_ARI:
                            pred_cond.append(p)
                    cond_unit_2[-1] = True if cond_unit_2[-1] else False

                if label_ari != 0:
                    for _comp in pred[10:] :
                        _, l = _comp
                        if PRED_WHERE_ARI:
                            gold_cond.append(l[i][j])

                pred_set.add('-'.join([str(v) for v in pred_cond]))
                gold_set.add('-'.join([str(v) for v in gold_cond]))
                pred_cond_units.append(where_having_unit)

            # if len(pred_set)==len(gold_set) and len(pred_set-gold_set)==0 :
            #     cond_acc.append(True)
            # else :
            #     cond_acc.append(False)
            #
            # clause_acc.append(
            #     num_acc[-1] and cond_acc[-1]
            # )

            if len(pred_set)==len(gold_set) and len(pred_set-gold_set)==0 :
                clause_cond_acc = True
            else :
                clause_cond_acc = False

            clause_acc.append(
                num_acc[-1] and clause_cond_acc
            )
            # print(utt_ids[i])
            # print(pred_and_or_list)
            # print(pred_cond_units)
            # print()

        results[clause_type+"conj_acc"] = np.array(conj_acc)
        results[clause_type+"_not_cond_acc"] = np.array(not_cond_acc)
        results[clause_type+"_cond_acc"] = np.array(cond_acc)
        results[clause_type+"_nest_1_acc"] = np.array(nest_1_acc)
        results[clause_type+"_nest_2_acc"] = np.array(nest_2_acc)
        results[clause_type+"_ari_acc"] = np.array(ari_acc)
        results[clause_type+"_agg1_acc"] = np.array(agg1_acc)
        results[clause_type+"_col1_acc"] = np.array(col1_acc)
        results[clause_type+"_dist1_acc"] = np.array(dist1_acc)
        results[clause_type+"_clause"] = np.array(clause_acc)


    if "where" in preds :
        clause_type = "where"
        eval_clause(clause_type, preds, results, table_map, used_tables, sqls, where_clause=True)
        sample_pred *= results[clause_type + "_clause"]

    if "having" in preds :
        clause_type = "having"
        eval_clause(clause_type, preds, results, table_map, used_tables, sqls)
        sample_pred *= results[clause_type + "_clause"]
    ##############################################################
    # measure table clause accuracy
    if "table" in preds :
        (table_id_pred, table_id_label), (table_num_pred, table_num_label) =\
            preds["table"]
        table_id_label = table_id_label[:, :table_id_pred.shape[1]]
        table_num_pred = np.argmax(table_num_pred, axis=-1)
        results["table_num"] = table_num_pred==table_num_label 

        table_id_acc = []
        for i in range(batch_size) :
            gold_tables = np.argsort(table_id_label[i])[-table_num_label[i]-1:]

            if tables:
                """
                    evaluation process
                    new sql table prediction 
                    only use existed table_id in DB during prediction
                """
                table_dict = {
                    table_index: table_name for table_index, table_name in enumerate(tables[i]["table_names_original"])
                }
                pred_tables = []
                for table_id in np.argsort(table_id_pred[i])[::-1]:
                    if table_id in table_dict:
                        pred_tables.append(table_id)
                        if len(pred_tables) == table_num_pred[i] + 1:
                            break
                sqls[i]["table"] = pred_tables
            else:
                """
                    validation process
                    origin sql table prediction
                """
                # prediction 한 column 해당되는 table id를 강제적으로 추가하여
                # table prediction 에러를 최소화
                pred_tables = np.argsort(table_id_pred[i])[-table_num_pred[i]-1:]
                pred_tables = set(pred_tables).union(set([t_id for t_id in used_tables[i] if t_id!=-1]))
                sqls[i]["table"].extend(list(pred_tables))

                #
                # if len(gold_tables) > 1:
                #     print(utt_ids[i])
                #     print(used_tables[i])
                #     print(gold_tables)
                #     print(list(pred_tables))
                #     print()

            if len(gold_tables)==len(pred_tables) and len(set(gold_tables)-set(pred_tables))==0 :
                table_id_acc.append(True)
            else :
                table_id_acc.append(False)
        results["table_id"] = np.array(table_id_acc)
        sample_pred *= results["table_id"]
    ##############################################################

    results["final_sample"] = sample_pred
    decode_sql_list = decode_sql(sqls, tables, utt_ids)

    return results, decode_sql_list


def decode_sql(sqls, tables, utt_ids):
    def decode_col_units(col_units, clause="select"):
        """
            col_units = [
                col_unit = [
                    uni_op
                    [                   -> cond_unit_1
                        col_agg,
                        col_index,
                        is_dist
                    ],
                    [                   -> cond_unit_2
                        col_agg,
                        col_index,
                        is_dist
                    ]
                ],
                ...
            ]
        """
        decode_col_str = ""
        col_num = len(col_units)
        for _j in range(col_num):
            col_unit = col_units[_j]
            if clause == "select":
                col_agg = col_unit[0]
                unit_op, cond_unit_1, cond_unit_2 = col_unit[1]
            else:
                col_agg = 0
                unit_op, cond_unit_1, cond_unit_2 = col_unit

            _unit_str = decode_select_cond_unit(cond_unit_1)
            if unit_op:
                _unit_str += Data.ARI[unit_op] + decode_select_cond_unit(cond_unit_2)

            if col_agg:
                # _unit_str = " " + Data.AGG[col_agg] + "(" + _unit_str + ")" + " "
                _unit_str = " " + Data.AGG[col_agg] + "(" + _unit_str.strip() + ")" + " "

            _is_last_col = (_j == col_num - 1)
            if not _is_last_col:
                # _unit_str += ","
                if _unit_str[-1] != " ":
                    print("\nERROR) DECODE COLUMN NAMES")
                    print("utt_id -", utt_ids[i])
                    print("_unit_str -", "'" + _unit_str + "'")
                    print("_unit_str[-1] -", "'" + _unit_str[-1] + "'")
                    print("_unit_str[:-1] -", "'" + _unit_str[:-1] + "'", "\n\n")
                    exit(-1)
                _unit_str = _unit_str[:-1] + ","

            decode_col_str += _unit_str

        return decode_col_str

    def decode_select_cond_unit(cond_unit):
        """
            Use : select column / group by column / order by column
            cond_unit = [
                col_agg,
                col_index,
                is_dist
            ]
        """
        agg, col, dist = cond_unit

        if DECODE_ERROR_IGNORE:
            try:
                if col == 0:
                    pred_col_name = "*"
                else:
                    pred_col_name = column_dict[col]
            except KeyError:
                pred_col_name = DECODE_ERROR_STR
        else:
            if col == 0:
                pred_col_name = "*"
            else:
                pred_col_name = column_dict[col]

        if agg:
            if dist:
                # DISTINCT COUNT(col)
                cond_str = "DISTINCT " + Data.AGG[agg] + "(" + pred_col_name + ")"
            else:
                # COUNT(col)
                cond_str = Data.AGG[agg] + "(" + pred_col_name + ")"
        else:
            if dist:
                # DISTINCT col
                cond_str = "DISTINCT " + pred_col_name
            else:
                # col
                cond_str = pred_col_name

        return " " + cond_str + " "

    def decode_where_cond_units(target_sql, where_clause=True):
        decode_str = decode_where_cond_unit(target_sql["cond_units"][0])
        if target_sql["and_or"]:
            for _cond_unit, _and_or in zip(target_sql["cond_units"][1:], target_sql["and_or"]):
                decode_str += _and_or.upper()
                decode_str += decode_where_cond_unit(_cond_unit, where_clause=where_clause)

        return decode_str

    def decode_where_cond_unit(_cond_unit, where_clause=True):
        """
            Use : where condition / having condition
        """
        not_op, op, cond_units, nest_1, nest_2 = _cond_unit
        not_op_str = "NOT " if not_op else ""
        val_1 = " " + VALUE_STR + " "
        val_2 = " " + VALUE_STR + " "

        if nest_1:
            if where_clause:
                val_1 = " " + WHERE_NESTED_STR + " "
            else:
                val_1 = " " + HAVING_NESTED_STR + " "

        if nest_2:
            if where_clause:
                val_2 = " " + WHERE_NESTED_STR + " "
            else:
                val_2 = " " + HAVING_NESTED_STR + " "

        if op:
            where_cond_unit = decode_col_units([cond_units], clause="where") + not_op_str + Data.OPS[op].upper() + val_1
            if op == Data.OPS.index("between"):
                where_cond_unit += "AND" + val_2
        else:
            where_cond_unit = " " + DECODE_ERROR_STR + " "

        return where_cond_unit

    if not tables:
        """
            validation process
            not use sql decoding for early stopping
        """
        return list()

    decode_sql_list = []
    # flexible batch_size (if mask == 1)
    for i in range(len(sqls)):
        sql = sqls[i]
        column_dict = {
            col_index: column_name for col_index, column_name in enumerate(tables[i]["column_names_original"])
        }
        table_dict = {
            table_index: table_name for table_index, table_name in enumerate(tables[i]["table_names_original"])
        }
        decode_result = OrderedDict(
            {
                "select": "SELECT",
                "table": "",
                "where": "",
                "groupBy": "",
                "having": "",
                "orderBy": "",
                "limit": ""
            }
        )

        """
            table clause  (JOIN)
        """

        table_sql = sql["table"]
        decode_result["table"] = "FROM " + table_dict[table_sql[0]]

        for join_table_index in table_sql[1:]:
            decode_result["table"] += " JOIN " + table_dict[join_table_index]

        # print(table_sql)
        # print()
        # print(tables[i]["table_names_original"])
        # print()
        # print(decode_result["table"])
        # print()
        # exit(-1)

        """
            select clause
        """
        select_sql = sql["select"]
        # select_clause = decode_result["select"]

        if select_sql[0]:
            decode_result["select"] += " DISTINCT"

        decode_result["select"] += decode_col_units(select_sql[1])

        # print(utt_ids[i])
        # print(select_sql)
        # print(decode_result["select"])
        # print()

        """
            order by clause
        """
        order_sql = sql["orderBy"]
        if order_sql:
            decode_result["orderBy"] = "ORDER BY" + \
                                       decode_col_units(order_sql[1], clause="orderBy") + \
                                       Data.ORDER_MAP_REV[order_sql[0]]
            # print(utt_ids[i], order_sql)
            # print(decode_result["orderBy"])
            # print()

        """
            group by clause
        """
        group_sql = sql["groupBy"]
        if group_sql:
            decode_result["groupBy"] = "GROUP BY"

            grb_num = len(group_sql)
            for j in range(grb_num):
                unit_str = decode_select_cond_unit(group_sql[j])
                is_last_col = (j == grb_num - 1)
                if not is_last_col:
                    # unit_str += ","
                    if unit_str[-1] != " ":
                        print("\nERROR) DECODE COLUMN NAMES")
                        print("utt_id -", utt_ids[i])
                        print("_unit_str -", "'" + unit_str + "'")
                        print("_unit_str[-1] -", "'" + unit_str[-1] + "'")
                        print("_unit_str[:-1] -", "'" + unit_str[:-1] + "'", "\n\n")
                        exit(-1)
                    unit_str = unit_str[:-1] + ","

                decode_result["groupBy"] += unit_str
            # print(utt_ids[i], group_sql)
            # print(decode_result["groupBy"])
            # print()

        """
            limit clause
        """
        limit_sql = sql["limit"]
        if limit_sql:
            is_top1, pos = limit_sql

            if is_top1:
                decode_result["limit"] = "LIMIT 1"
            else:
                """
                   TODO: get VALUE (token position)
                """
                decode_result["limit"] = 'LIMIT ' + VALUE_STR
            # print(decode_result["limit"])
            # print()

        """
            where clause
        """
        where_sql = sql["where"]
        if where_sql["cond_units"]:
            decode_result["where"] = "WHERE" + decode_where_cond_units(where_sql)
            # print(utt_ids[i])
            # print(where_sql)
            # print(decode_result["where"])
            # print()
            # exit(-1)

        """
            having clause
        """
        having_sql = sql["having"]
        if having_sql["cond_units"]:
            decode_result["having"] = "HAVING" + decode_where_cond_units(having_sql, where_clause=False)
            # print(utt_ids[i])
            # print(having_sql)
            # print(decode_result["having"])
            # print()

        """
            concat sql
            IUEN -> SELECT -> TABLE -> WHERE -> GROUP BY -> HAVING -> ORDER BY -> LIMIT
        """
        decoded_concat_str = Data.IEUN_MAP_REV[sql["iuen"]]
        for _, decoded_str in decode_result.items():
            if decoded_str:
                decoded_concat_str += " " + decoded_str.strip()
        decode_sql_list.append(decoded_concat_str.strip())

        #
        # decoded_concat_str = str()
        # for _, decoded_str in decode_result.items():
        #     if decoded_str:
        #         decoded_concat_str += " " + decoded_str.strip()
        # decoded_concat_str = decoded_concat_str.strip()
        #
        # """
        #     Add IEU
        # """
        # decoded_concat_str = Data.IEUN_MAP_REV[sql["iuen"]] + " " + decoded_concat_str
        # decode_sql_list.append(decoded_concat_str.strip())

        # if order_sql:
        #     print(utt_ids[i])
        #     print(decoded_concat_str)
        #     print()

    return decode_sql_list


def _get_sub_sql_index_list(decode_tokens):
    sub_sql_index_list = []
    sub_sql_index_dict = {}

    for tok_index, tok in enumerate(decode_tokens):
        if tok == WHERE_NESTED_STR or tok == HAVING_NESTED_STR:
            sub_sql_index_list.append(tok_index)
        sub_sql_index_dict[tok_index] = tok

    # try:
    #     for tok_index, tok in enumerate(decode_tokens):
    #         if tok == WHERE_NESTED_STR or tok == HAVING_NESTED_STR:
    #             sub_sql_index_list.append(tok_index)
    #         sub_sql_index_dict[tok_index] = tok
    # except:
    #     print("decode_tokens -", decode_tokens)
    #     exit(-1)

    return sub_sql_index_list[::-1], sub_sql_index_dict


def _recursive_concat(reverse_decode_sql_list, depth):
    if depth >= len(reverse_decode_sql_list):
        """
            overflow decode string
        """
        return "", depth

    target_decode_result = reverse_decode_sql_list[depth]
    if depth == len(reverse_decode_sql_list) - 1:
        """
            last decode string
        """
        return target_decode_result, depth

    """
        reverse_decode_sql_list[depth:] -> remain decode string list
        len(reverse_decode_sql_list[depth:]) -> at least 1
    """
    decode_tokens = target_decode_result.split()
    reverse_sub_sql_index_list, sub_sql_index_dict = _get_sub_sql_index_list(decode_tokens)

    # if reverse_sub_sql_index_list:
    #     print()
    #     print(reverse_decode_sql_list)
    #     print()
    #     print(target_decode_result)
    #     print()

    # leave_count = len(reverse_decode_sql_list[depth:])
    # if len(reverse_sub_sql_index_list) > leave_count:
    #     reverse_sub_sql_index_list = reverse_sub_sql_index_list[:leave_count]
    if reverse_sub_sql_index_list:
        for sub_sql_index in reverse_sub_sql_index_list:
            """
                # SELECT A FROM TABLE WHERE B IN (sub-sql) AND C = "#VALUE#" AND D NOT IN (sub-sql)
                #                                    1                                        2
                ordering  2 -> 1
                
                if sub-sql has IEU
                
            """
            # prefix_str = " ".join(decode_tokens[:sub_sql_index])
            # if sub_sql_index == len(decode_tokens) - 1:
            #     suffix_str = ""
            # else:
            #     suffix_str = " ".join(decode_tokens[sub_sql_index + 1:])
            target_decode_result, depth = _recursive_concat(reverse_decode_sql_list, depth + 1)
            if target_decode_result:
                sub_sql_index_dict[sub_sql_index] = "(" + target_decode_result + ")"

            # print("prefix_str -", prefix_str)
            # print("suffix_str -", suffix_str, "\n")

        concat_sub_sql = " ".join([sub_sql_index_dict[tok_index] for tok_index in range(len(decode_tokens))])

        # print("-----------------------------------------")
        # print("len(reverse_decode_sql_list) -", len(reverse_decode_sql_list))
        # print("concat_sub_sql -", concat_sub_sql)
        # print("-----------------------------------------")
        # exit(-1)

        return concat_sub_sql.strip(), depth
    else:
        """
            case 1) next decode string has IEU
                return next depth and keep going
            case 2) next decode string has None
                return finish
        """
        next_target_decode_result = reverse_decode_sql_list[depth + 1]
        if next_target_decode_result.split()[0].lower() in Data.IEUN_MAP:
            # ["SELECT ~~~~"(NOT sub-sql), "UNION SELECT ~~~~"]
            return target_decode_result, depth
        else:
            # print()
            # print("next_target -", next_target_decode_result)
            # ["SELECT ~~~~"(NOT sub-sql), "SELECT ~~~~"]
            return target_decode_result, len(reverse_decode_sql_list)


def concat_decode_sql(decode_results, utt_ids):

    # if len(decode_results) != len(utt_ids):
    #     print(len(decode_results))
    #     print(len(utt_ids))
    #     exit(-1)
    concat_decode_results = []
    for i, decode_sql_list in enumerate(decode_results):
        # concat_decode_str = str()
        reverse_decode_sql_list = decode_sql_list[::-1]
        # if len(decode_sql_list) > 1:
        #     print(utt_ids[i])
        #     print("reverse_decode_sql_list -", reverse_decode_sql_list)
        #     print()

        depth = 0
        result_str = str()
        while depth < len(reverse_decode_sql_list):
            recursive_decode_str, depth = _recursive_concat(reverse_decode_sql_list, depth)
            result_str += recursive_decode_str + " "
            depth += 1

        concat_decode_results.append(result_str.strip())
        # if len(decode_sql_list) > 1:
        #     print()
        #     print("result_str -", result_str)
        #     print("\n\n")

    return concat_decode_results
