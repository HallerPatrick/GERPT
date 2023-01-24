import torch
from datasets.load import load_from_disk

from src.dataset import GenericDataModule
from src.dictionary import Dictionary

ds = load_from_disk("data/tokenized_data")
dic: Dictionary = torch.load("data/dict_data")

# for row in ds["train"]:
#     source = row["source"]
#
#     seq = source[0]
#     seq2 = source[1]
#
#     dic.print_sequence(seq, 1)
#     dic.print_sequence(seq2, 2)
#     exit()
#
#

module = GenericDataModule(ds, 4, 100)

module.prepare_data()
dl = module.train_dataloader()


# s = [[ 20,   2,   8,   7,   1,   7,   4,   2,   1,   6],
#     [  4,   2,   1,   2,   3,  17,  15,   6,  27,   1],
#     [  1,  25,   3,  23,   2,   1,   7,   4,   2,   1]
# ]
#
# for k in s:
#     dic.print_sequence(k, 1)
#
# exit()

for j, i in enumerate(dl):

    src = i[0]
    trgt = i[0]

    print(src.size())
    print(trgt.size())

    batch_one_one_gram = src[0, :, 0]
    batch_one_two_gram = src[1, :, 0]

    batch_one_one_gram_target = trgt[0, :, 0]
    batch_one_two_gram_target = trgt[1, :, 0]

    batch_two_one_gram = src[0, :, 1]
    batch_two_two_gram = src[1, :, 1]

    
    # batch_three_one_gram = src[0, :, 2]
    # batch_three_two_gram = src[1, :, 2]
    #
    # batch_four_one_gram = src[0, :, 3]
    # batch_four_two_gram = src[1, :, 3]
    
    dic.print_sequence(batch_one_one_gram, 1)
    dic.print_sequence(batch_one_two_gram, 2)

    print("~"*90)
    dic.print_sequence(batch_one_one_gram_target, 1)
    dic.print_sequence(batch_one_two_gram_target, 2)
    print("-"*90)
    dic.print_sequence(batch_two_one_gram, 1)
    dic.print_sequence(batch_two_two_gram, 2)
    print("-"*90)
    # dic.print_sequence(batch_three_one_gram, 1)
    # dic.print_sequence(batch_three_two_gram, 2)
    # print("-"*90)
    # dic.print_sequence(batch_four_one_gram, 1)
    # dic.print_sequence(batch_four_two_gram, 2)
    print("="*90)

    # if j == 2:
    #     exit()

