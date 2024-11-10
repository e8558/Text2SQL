import transformers
# from transformers import BertTokenizer
# from kobert.tokenization_kobert import KoBertTokenizer
# from kobert import get_pytorch_kobert_model


def get_tokenizer(config) :
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.pretrained_model
        , cache_dir="cache"
    )
    # if config.pretrained_model == "monologg/kobert":
    #     # print("\n\n=== KoBertTokenizer ===\n\n")
    #     # tokenizer = KoBertTokenizer.from_pretrained(
    #     #     config.pretrained_model
    #     #     , cache_dir="cache"
    #     #     )
    #     _, tokenizer = get_pytorch_kobert_model()
    #
    #     # tokenizer.cls_id = 2
    #     # tokenizer.sep_id = 3
    #     # tokenizer.pad_id = 1
    # else:
    #     # print("\n\n=== AutoTokenizer ===\n\n")
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         config.pretrained_model
    #         , cache_dir="cache"
    #         )
    # tokenizer = BertTokenizer.from_pretrained(
    #     config.pretrained_model
    #     , cache_dir="cache"
    #     )

    # tokenizer.cls_id = tokenizer._convert_token_to_id("[CLS]")
    # tokenizer.sep_id = tokenizer._convert_token_to_id("[SEP]")
    # tokenizer.pad_id = tokenizer._convert_token_to_id("[PAD]")
    tokenizer.cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
    tokenizer.sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    tokenizer.pad_id = tokenizer.convert_tokens_to_ids("[PAD]")

    return tokenizer