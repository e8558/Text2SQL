import os
import argparse
import time
import numpy as np
import pandas as pd

import json
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from collections import OrderedDict
import datetime
import random
import warnings
import torch.backends.cudnn as cudnn

import model_ko as Model
import data_ko as Data
import tokenization
import utils_ko
import optimization

from progressbar import printProgressBar, cal_running_avg_loss
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.simplefilter("ignore", UserWarning)

ROUND_NUM = 3


def get_loader(config, data, batch_size, train=True) :
    "make data loaders"
    data = TensorDataset(torch.from_numpy(data).to(config.device))
    if train :
        sampler = RandomSampler(data)
    else :
        sampler = SequentialSampler(data)
    return DataLoader(data
        , batch_size=batch_size
        , shuffle=(sampler is None)
        , sampler=sampler, drop_last=train), sampler
    # if not test:
    #     return DataLoader(data
    #         , batch_size=batch_size
    #         , shuffle=(sampler is None)
    #         , sampler=sampler, drop_last=True), sampler
    # else:
    #     return DataLoader(data
    #         , batch_size=batch_size
    #         , shuffle=(sampler is None)
    #         , sampler=sampler), sampler


def convert_to_tensor(elements, device="cpu") :
    arr = []
    for e in elements :
        arr.append(e)
    arr = torch.from_numpy(np.array(arr, dtype=np.int8)).to(device)
    """
        for DataLoader(drop_last = False) -> erase squeeze  
    """
    # if arr.size(-1)==1 :
    #     arr = arr.squeeze(-1)
    return arr


def make_inputs(config, samples, device="cpu") :
    batch_size = len(samples)
    sample = []
    q_len = []
    c_len = []
    t_len = []
    label = []
    table_map = []

    max_sql_num = 0
    # tuple -> list, max sql num 구함
    for sample_ in samples :
        sample.append(sample_["tokens"])
        q_len.append(sample_["q_len"])
        c_len.append(sample_["c_len"])
        t_len.append(sample_["t_len"])
        label.append(sample_["label"])
        max_sql_num = max(max_sql_num, len(sample_["label"]))
        table_map.append(sample_["table_map"])

    # sql_mask_mat for BERT model  ( Y_mask --> BERT )
    sql_mask = np.zeros((batch_size, max_sql_num))
    for i in range(batch_size) :
        sql_mask[i, :len(label[i])] = 1
        for _ in range(max_sql_num-len(label[i])) :
            label[i] += (label[i][0], )

    # all_labels_mat for BERT model  ( Y --> BERT )
    all_labels = tuple()
    for i in range(max_sql_num) :
        all_labels_ = tuple()
        for j in range(len(label[0][0])) :
            all_labels_ += (
                    convert_to_tensor([l[i][j] for l in label], device=device)
                , )
        all_labels += (all_labels_, )

    x, mask = utils_ko.pad_sequence(sample, max_seq=config.max_seq)
    x = torch.from_numpy(x).long().to(device)
    mask = torch.from_numpy(mask).long().to(device)
    sql_mask = torch.from_numpy(sql_mask).long().to(device)
    table_map = np.array(table_map)

    # print(len(x), len(mask), len(q_len), len(c_len), len(t_len), len(sql_mask), len(all_labels), len(table_map))
    return (x, mask, q_len, c_len, t_len, sql_mask, all_labels, table_map)


# def run_epoch(config, epoch_num, model, optimizer, samples, loader, train=True, do_show=False) :
def run_epoch(config, epoch_num, model, optimizer, samples, loader, tables=None, train=True, do_show=False,
              save_decode=False, show_progress=True):
    start_time = time.time()
    current_dt = datetime.datetime.now()
    batch_num = len(loader)
    results = dict()
    avg_loss = 0.0
    prefix = " train" if train else " valid"
    prefix += " epoch " + str(epoch_num + 1).rjust(2) + "   "

    if save_decode:
        save_decode_io = open(os.path.join(config.save_ckpt, "predictions.txt"), "w")
    else:
        save_decode_io = None

    # print(tables[0])
    for i, batch in enumerate(loader) :
        idx = batch[0].cpu().numpy().tolist()
        inputs = make_inputs(config, [samples[_i] for _i in idx], config.device)
        inputs += (train, )
        if tables:
            target_tables = [tables[samples[_i]["db_id"]] for _i in idx]
            inputs += (target_tables, )

        if not train:
            utt_ids = [samples[_i]["utt_id"] for _i in idx]
            inputs += (utt_ids, )

        loss, result, decode_result = model(*inputs)
        for k, v in result.items() :
            if k not in results :
                results[k] = []
            results[k].extend(v)
        avg_loss = round(cal_running_avg_loss(loss.item(), avg_loss), 4)
        if train :
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        if show_progress:
            printProgressBar(
                i+1, batch_num, start_time, prefix=prefix, suffix=" loss {} ".format(avg_loss), length=40,
                start_time_str="[" + current_dt.strftime("%m-%d %H:%M:%S") + "]"
            )

        if save_decode_io and not train:
            for decode_sql in decode_result:
                save_decode_io.write(decode_sql + "\n")

    final_score = 0
    if not train :
        df = []
        for k, v in results.items() :
            if len(v)!=0 :
                score = round(sum(v)/len(v), 4)
            else :
                score = " - "
            if k=="final_result" :
                final_score = score
                # print(" EM -", final_score * 100)
            df.append([k, len(v), score])
            # df.append([k, score])
        df = pd.DataFrame(df)
        df.columns = ["sql_part", "sample_num", "accuracy"]
        # df.columns = ["sql_part", "  Exact Set Match"]

        if do_show:
            print("\n")
            print(df)
            print("\n")

        # if final_score > config.valid_score :
        #     config.valid_score = final_score
        #     if do_show:
        #         print("\n")
        #         print(df)
        #         print("\n")

    return avg_loss, final_score, results


def evaluate(config, model, optimizer, dev_dataset, tables=None, do_show=False, save_decode=False):
    dev_loader, _ = get_loader(config
                               , np.array([i for i in range(len(dev_dataset))])
                               , config.batch_size
                               , train=False
                               )
    # for k, v in dev_dataset[0].items():
    #     print(k, v)

    current_dt = datetime.datetime.now()
    print('[{0}] Start Evaluate:'.format(current_dt.strftime("%m-%d %H:%M:%S")))

    with torch.no_grad() :
        # _, _, results = run_epoch(
        #     config, 0, model, optimizer, dev_dataset, dev_loader, tables=tables, show_progress=False,
        #     train=False, do_show=False
        # )
        _, _, results = run_epoch(
            config, 0, model, optimizer, dev_dataset, dev_loader, tables=tables, show_progress=False,
            train=False, do_show=do_show, save_decode=save_decode
        )
        current_dt = datetime.datetime.now()
        print('[{0}] End Evaluate: Model predictions saved to {1}'.format(
            current_dt.strftime("%m-%d %H:%M:%S"),
            os.path.join(config.save_ckpt, "predictions.txt"))
        )

    evaluate_dict = OrderedDict()
    evaluate_dict["count"] = str(len(dev_dataset))
    for k in ["select_clause", "where_clause", "grb_clause", "ord_clause", "having_clause", "final_result"]:
        v = results[k]
        if len(v) != 0:
            score = round(sum(v)/len(v), ROUND_NUM)
        else:
            score = " - "
        evaluate_dict[k] = score

    return evaluate_dict


def evaluate_all(config, model, optimizer, dataset, tables):
    current_dt = datetime.datetime.now()
    print('[{0}] Start Evaluate:\n'.format(current_dt.strftime("%m-%d %H:%M:%S")))
    model.eval()

    hardness_list = ["easy", "medium", "hard", "extra"]
    dev_set = {hardness: list() for hardness in hardness_list}
    for sample in dataset["dev"]:
        hardness = "extra" if sample["hardness"].startswith("extra") else sample["hardness"]
        dev_set[hardness].append(sample)

    result_dict = dict()
    for hardness, dev_set_list in dev_set.items():
        # print()
        # print(hardness, "-", len(dev_set_list))
        result_dict[hardness] = evaluate(config, model, optimizer, dev_set_list, do_show=True)

    result_dict["total"] = evaluate(
        config, model, optimizer, dataset["dev"], tables=tables, do_show=True, save_decode=True
    )
    df = []
    for k, _ in result_dict["total"].items():
        line = [k]
        for _, evaluate_dict in result_dict.items():
            line.append(evaluate_dict[k])
        df.append(line)

    df = pd.DataFrame(df)
    df.columns = [''] + [hardness for hardness in result_dict]
    # print(df)

    current_dt = datetime.datetime.now()
    print('\n[{0}] End Evaluate:\n'.format(current_dt.strftime("%m-%d %H:%M:%S")))


def add_config(config):
    # class DotDict(dict):
    #     """dot.notation access to dictionary attributes"""
    #     __getattr__ = dict.get
    #     __setattr__ = dict.__setitem__
    #     __delattr__ = dict.__delitem__

    with open("config.json", "r") as f:
        config_file = json.load(f)

    for k, v in config_file[config.lang].items():
        config.__setattr__(k, v)


def main(config) :
    add_config(config)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    if "large" in config.pretrained_model :
        config.hidden_size = 1024
    else :
        config.hidden_size = 768

    tokenizer = tokenization.get_tokenizer(config)
    # config.cls_id = tokenizer._convert_token_to_id("[CLS]")
    # config.sep_id = tokenizer._convert_token_to_id("[SEP]")
    # config.msk_id = tokenizer._convert_token_to_id("[MASK]")

    # data load
    current_dt = datetime.datetime.now()
    print('[{0}] Featuring Data:'.format(current_dt.strftime("%m-%d %H:%M:%S")))
    dataset, tables, train_tables, dev_tables = Data.load_spider(config, tokenizer)

    # exit(-1)
    def save(save_path, prefix, model, epoch) :
        "saving model weights"
        model_to_save = model.module if hasattr(model,
                        'module') else model  # Only save the model it-self
        save_dict = {
            'config' : config
            , 'model': model_to_save.state_dict()
        }
        torch.save(save_dict, os.path.join(save_path, prefix))

    def load(path, model) :
        current_dt = datetime.datetime.now()
        print('\n[{0}] loading model from: {1}\n'.format(current_dt.strftime("%m-%d %H:%M:%S"), path))
        load_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model.load_state_dict(load_dict['model'])

    model = Model.Model(config)

    # if config.save_ckpt is not None :
    if not config.train:
        load(os.path.join(config.save_ckpt, config.ckpt_name), model)

    model = model.to(int(config.device))
    # model_state_dict = model.state_dict()

    if config.train:
        print("\nbatch size -", config.batch_size)
        print("learning rate -", config.learning_rate)
        print("weight decay -", config.weight_decay)
        print("patience -", config.patience, "\n")
    # exit(-1)
    if config.weight_decay:
        optimizer = optimization.AdamW(optimization.get_optim_params(config, model)
                , lr=config.learning_rate
                , eps=config.optimizer_epsilon
                , weight_decay=config.weight_decay)
    else:
        optimizer = optimization.AdamW(optimization.get_optim_params(config, model)
                , lr=config.learning_rate
                , eps=config.optimizer_epsilon)

    config.valid_score = 0

    # if config.save_ckpt is not None :
    if config.train:
        """
            make directory for saving
        """
        if os.path.exists(config.save_ckpt):
            print("\n\n\tERROR)", config.save_ckpt, "already existed", "\n\n")
            exit(-1)
        os.mkdir(config.save_ckpt)

        """
            initialize train, valid loader
        """
        patience = 0
        best_score = 0
        train_loader, _ = get_loader(config
                                     , np.array([i for i in range(len(dataset["train"]))])
                                     , config.batch_size
                                     )
        valid_loader, _ = get_loader(config
                                     , np.array([i for i in range(len(dataset["valid"]))])
                                     , config.batch_size
                                     , train=False
                                     )
        for e in range(config.epoch):
            model.train()
            run_epoch(config, e, model, optimizer, dataset["train"], train_loader)

            model.eval()
            with torch.no_grad():
                _, final_score, _ = run_epoch(config, e, model, optimizer, dataset["valid"], valid_loader, train=False)
                if final_score > best_score:
                    best_score = final_score
                    patience = 0
                    current_dt = datetime.datetime.now()
                    print('[{0}]  Save checkpoint: valid score - {1}'.format(current_dt.strftime("%m-%d %H:%M:%S"), final_score))
                    save(config.save_ckpt, config.ckpt_name, model, e)
                else:
                    patience += 1

                if patience >= config.patience:
                    current_dt = datetime.datetime.now()
                    print('[{0}] End Training: Early Stopping\n'.format(current_dt.strftime("%m-%d %H:%M:%S")))
                    break

            print("")

    else:
        """
            write sql file for spider official evaluate script 
        """
        with open(os.path.join(config.save_ckpt, "dev_gold_parsed.sql"), "w") as f:
            for sample in dataset["dev"]:
                f.write(" ".join(sample["query"].split("\t")) + "\t" + sample["db_id"] + "\n")

        with open(os.path.join(config.save_ckpt, "dev_gold_parsed_info.txt"), "w") as f:
            for sample in dataset["dev"]:
                f.write(sample["utt_id"].strip() + "\n")

        evaluate(config, model, optimizer, dataset["dev"], tables=tables, do_show=False, save_decode=True)
        # evaluate_all(config, model, optimizer, dataset, tables)


if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--random_seed", default=42, type=int, help="random state (seed)")
    parser.add_argument("--epoch", default=10000, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--max_seq", default=512, type=int, help="maximum length of long text sequence i.e. pretrain segment")

    parser.add_argument("--optimizer_epsilon", default=1e-8, type=float, help="epsilon value for optimzier")
    parser.add_argument("--weight_decay", default=0.9, type=float, help="L2 weight decay for optimizer")
    parser.add_argument("--dropout", default=0.1, type=float, help="probability of dropout layer")

    parser.add_argument("--lang", help="select language\n", choices=["ko", "en"], default="ko")
    parser.add_argument("--train", action="store_true", help="scatter paragraph into paragraphs")
    parser.add_argument("--save_ckpt", default="save_model/test", type=str)
    parser.add_argument("--ckpt_name", default="best_ckpt", type=str)
    parser.add_argument("--device", default=2, type=int)
    args = parser.parse_args()

    main(args)
