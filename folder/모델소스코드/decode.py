import os
import argparse
import time
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import random
import warnings
import torch.backends.cudnn as cudnn

import model_ko as Model
import data_ko as Data
import tokenization
import utils
import optimization

from progressbar import printProgressBar, cal_running_avg_loss
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.simplefilter("ignore", UserWarning)


def get_loader(config, data, batch_size, train=True):
    "make data loaders"
    data = TensorDataset(torch.from_numpy(data).to(config.device))
    if train:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    return DataLoader(data
        , batch_size=batch_size
        , shuffle=(sampler is None)
        , sampler=sampler, drop_last=True), sampler


def convert_to_tensor(elements, device="cpu"):
    arr = []
    for e in elements:
        arr.append(e)
    arr = torch.from_numpy(np.array(arr, dtype=np.int8)).to(device)

    # if arr.size(-1) == 1:
    #     arr = arr.squeeze(-1)
    return arr


def make_inputs(config, samples, device="cpu"):
    batch_size = len(samples)
    sample = []
    q_len = []
    c_len = []
    t_len = []
    label = []
    table_map = []

    max_sql_num = 0
    # tuple -> list, max sql num 구함
    for sample_ in samples:
        sample.append(sample_["tokens"])
        q_len.append(sample_["q_len"])
        c_len.append(sample_["c_len"])
        t_len.append(sample_["t_len"])
        label.append(sample_["label"])
        max_sql_num = max(max_sql_num, len(sample_["label"]))
        table_map.append(sample_["table_map"])

    # sql_mask_mat for BERT model  ( Y_mask --> BERT )
    sql_mask = np.zeros((batch_size, max_sql_num))
    for i in range(batch_size):
        sql_mask[i, :len(label[i])] = 1
        for _ in range(max_sql_num - len(label[i])):
            label[i] += (label[i][0],)

    # all_labels_mat for BERT model  ( Y --> BERT )
    all_labels = tuple()
    for i in range(max_sql_num):
        all_labels_ = tuple()
        for j in range(len(label[0][0])):
            all_labels_ += (
                convert_to_tensor([l[i][j] for l in label], device=device)
                ,)
        all_labels += (all_labels_,)

    x, mask = utils.pad_sequence(sample, max_seq=config.max_seq)
    x = torch.from_numpy(x).long().to(device)
    mask = torch.from_numpy(mask).long().to(device)
    sql_mask = torch.from_numpy(sql_mask).long().to(device)
    table_map = np.array(table_map)

    # print(len(x), len(mask), len(q_len), len(c_len), len(t_len), len(sql_mask), len(all_labels), len(table_map))
    return (x, mask, q_len, c_len, t_len, sql_mask, all_labels, table_map)


# def run_epoch(config, epoch_num, model, optimizer, samples, loader, train=True, do_show=False) :
def run_epoch(config, epoch_num, model, optimizer, samples, loader, tables=None, train=True, do_show=False,
              show_progress=True):
    start_time = time.time()
    batch_num = len(loader)
    results = dict()
    avg_loss = 0.0
    prefix = " train" if train else " valid"
    prefix += " epoch " + str(epoch_num + 1).rjust(2) + "   "

    for i, batch in enumerate(loader):
        idx = batch[0].cpu().numpy().tolist()
        inputs = make_inputs(config, [samples[i] for i in idx], config.device)
        inputs += (train,)
        if tables:
            target_tables = [tables[samples[_i]["db_id"]] for _i in idx]
            inputs += (target_tables,)
        loss, result = model(*inputs)
        for k, v in result.items():
            if k not in results:
                results[k] = []
            results[k].extend(v)
        avg_loss = round(cal_running_avg_loss(loss.item(), avg_loss), 4)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        if show_progress:
            printProgressBar(i + 1, batch_num, start_time, prefix=prefix, suffix=" loss {} ".format(avg_loss),
                             length=40)

    final_score = 0
    if not train:
        df = []
        for k, v in results.items():
            if len(v) != 0:
                score = round(sum(v) / len(v), 4)
            else:
                score = " - "
            if k == "final_result":
                final_score = score
                # print(" EM -", final_score * 100)
            df.append([k, len(v), score])
            # df.append([k, score])
        df = pd.DataFrame(df)
        df.columns = ["sql_part", "sample_num", "accuracy"]
        # df.columns = ["sql_part", "  Exact Set Match"]

        if final_score > config.valid_score:
            config.valid_score = final_score
            if do_show:
                print("\n")
                print(df)
                print("\n")

    return avg_loss, final_score


def main(config):
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    if "large" in config.pretrained_model:
        config.hidden_size = 1024
    else:
        config.hidden_size = 768

    tokenizer = tokenization.get_tokenizer(config)
    batch_size = 1

    # data load
    dataset, tables, train_tables, dev_tables = Data.load_spider(config, tokenizer)


    train_loader, _ = get_loader(config
                                 , np.array([i for i in range(len(dataset["train"]))])
                                 , config.batch_size
                                 )
    valid_loader, _ = get_loader(config
                                 , np.array([i for i in range(len(dataset["valid"]))])
                                 , config.batch_size
                                 , train=False
                                 )
    dev_loader, _ = get_loader(config
                               , np.array([i for i in range(len(dataset["dev"]))])
                               # , config.batch_size
                               , batch_size
                               , train=False
                               )

    def load(path, model) :
        print("loading model from ", path, "\n")
        load_dict = torch.load(path, mapz_location=lambda storage, loc: storage)
        model.load_state_dict(load_dict['model'])

    model = Model.Model(config)

    if config.save_ckpt is not None :
        load(config.save_ckpt, model)

    model = model.to(int(config.device))
    model.eval()
    len_dev = len(dataset["dev"])

    def idx_tensor(t, idx):
        return t[idx == 1]


    for i in range(0, len_dev, batch_size):
        if i + batch_size > len_dev:
            samples = dataset["dev"][i:]
            target_tables = [tables[samples[_i]["db_id"]] for _i in range(len_dev - i)]
        else:
            samples = dataset["dev"][i:i+batch_size]
            target_tables = [tables[samples[_i]["db_id"]] for _i in range(batch_size)]
        inputs = make_inputs(config, samples, config.device)
        loss, result = model(*inputs)

        exit(-1)
        # for k, v in result.items():
        #     if k not in results:
        #         results[k] = []
        #     results[k].extend(v)

        label = []

        # tuple -> list, max sql num 구함
        for sample_ in samples:
            """
                label[i][j]
                i -> sql_num for spc
                j -> samples
            """
            label.append(sample_["label"])
        sql_mask = inputs[5]
        all_labels = inputs[6]
        max_sql_num = len(all_labels)
        for _i in range(max_sql_num):
            idx = sql_mask[:, _i].view(-1)
            labels = all_labels[_i]

            """
                indexed_labels[i][j]
                i -> # of class in 1 label
                j -> batch_size
            """
            indexed_labels = tuple()

            for l in labels:
                indexed_labels += (idx_tensor(l, idx),)

            print(label[_i][0])
            print(len(label[_i][0]))
            print("\n")
            print(indexed_labels)
            print(len(indexed_labels))
            print("\n")

        print(len(label), len(all_labels))
        print()
        exit(-1)



    results = dict()
    avg_loss = 0.0
    for sample in dataset["dev"]:
        table = tables[sample["db_id"]]

        inputs = make_inputs(config, [sample], config.device)
        loss, result = model(*inputs)
        print(loss)


    final_score = 0
    df = []
    for k, v in results.items() :
        if len(v)!=0 :
            score = round(sum(v)/len(v), 4)
        else :
            score = " - "
        if k=="final_result" :
            final_score = score
        df.append([k, len(v), score])

    df = pd.DataFrame(df)
    df.columns = ["sql_part", "sample_num", "accuracy"]

    if final_score > config.valid_score :
        print("\n")
        print(df)
        print("\n")


    # for i in range(0, len_dev, batch_size):
    #     if i + batch_size > len_dev:
    #         samples = dataset["dev"][i:]
    #         target_tables = [tables[samples[_i]["db_id"]] for _i in range(len_dev - i)]
    #     else:
    #         samples = dataset["dev"][i:i+batch_size]
    #         target_tables = [tables[samples[_i]["db_id"]] for _i in range(batch_size)]
    #     inputs = make_inputs(config, samples, config.device)
    #     # loss, result = model(*inputs)
    #     # for k, v in result.items():
    #     #     if k not in results:
    #     #         results[k] = []
    #     #     results[k].extend(v)
    #
    #     label = []
    #
    #     # tuple -> list, max sql num 구함
    #     for sample_ in samples:
    #         """
    #             label[i][j]
    #             i -> sql_num for spc
    #             j -> samples
    #         """
    #         label.append(sample_["label"])
    #     sql_mask = inputs[5]
    #     all_labels = inputs[6]
    #     max_sql_num = len(all_labels)
    #     for _i in range(max_sql_num):
    #         idx = sql_mask[:, _i].view(-1)
    #         labels = all_labels[_i]
    #
    #         """
    #             indexed_labels[i][j]
    #             i -> # of class in 1 label
    #             j -> batch_size
    #         """
    #         indexed_labels = tuple()
    #
    #         for l in labels:
    #             indexed_labels += (idx_tensor(l, idx),)
    #
    #         print(label[_i][0])
    #         print(len(label[_i][0]))
    #         print("\n")
    #         print(indexed_labels)
    #         print(len(indexed_labels))
    #         print("\n")
    #
    #     print(len(label), len(all_labels))
    #     print()
    #     exit(-1)
    #

    # final_score = 0
    # df = []
    # for k, v in results.items() :
    #     if len(v)!=0 :
    #         score = round(sum(v)/len(v), 4)
    #     else :
    #         score = " - "
    #     if k=="final_result" :
    #         final_score = score
    #     df.append([k, len(v), score])
    #
    # df = pd.DataFrame(df)
    # df.columns = ["sql_part", "sample_num", "accuracy"]
    #
    # if final_score > config.valid_score :
    #     print("\n")
    #     print(df)
    #     print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--random_seed", default=42, type=int, help="random state (seed)")
    parser.add_argument("--epoch", default=10000, type=int, help="number of epochs")
    # parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--max_seq", default=512, type=int,
                        help="maximum length of long text sequence i.e. pretrain segment")

    parser.add_argument("--optimizer_epsilon", default=1e-8, type=float, help="epsilon value for optimzier")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="L2 weight decay for optimizer")
    parser.add_argument("--dropout", default=0.1, type=float, help="probability of dropout layer")

    parser.add_argument("--pretrained_model"
                        # , default="skt/kobert-base-v1"
                        , default="monologg/koelectra-base-v3-discriminator"
                        # , default="bert-base-multilingual-uncased"
                        , type=str)
    parser.add_argument("--pkl_name"
                        , default="pkl/koelectra-base-v3-discriminator/10_data_220921.pkl"
                        , type=str)
    parser.add_argument("--data_path"
                        , default="data/spider"
                        , type=str)
    parser.add_argument("--save_ckpt"
                        # , default="save_model/best_ckpt"
                        , default=None
                        , type=str)
    parser.add_argument("--device", default=2, type=int)
    args = parser.parse_args()

    main(args)
