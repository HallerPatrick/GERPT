import unittest

import torch

from src import utils
from src.dictionary import Dictionary
from src.models.ngme import n_hot
from src.models.ngme import NGramsEmbedding


class TestNGME(unittest.TestCase):
    def test_pack_tensor(self):
        ngram_sequences = [
            [1, 2, 3],
            [4, 5, 6],
        ]

        ngram_sequences = torch.tensor(ngram_sequences)

        packed_tensor = utils.pack_tensor(ngram_sequences)

        size = list(packed_tensor.size())

        self.assertEqual(size, [3])

    def test_packing_unpacking(self):

        # 3-gram sequence of 3 tokens
        ngram_sequences = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        tensor = torch.tensor(ngram_sequences)

        packed_sequence = utils.pack_tensor(tensor)

        unpacked = utils.unpack_tensor(packed_sequence)

        self.assertListEqual(ngram_sequences, unpacked.tolist())

    def test_tokenizing(self):

        dictionary = Dictionary(2, 10000, 10000, False, "dense")

        dictionary.add_ngram("<unk>", 1)
        dictionary.add_ngram("<pad>", 1)
        dictionary.add_ngram("a", 1)
        dictionary.add_ngram("b", 1)
        dictionary.add_ngram("c", 1)
        dictionary.add_ngram("d", 1)

        dictionary.add_ngram("<unk>", 2)
        dictionary.add_ngram("<pad>", 2)
        dictionary.add_ngram("ab", 2)
        dictionary.add_ngram("bc", 2)
        dictionary.add_ngram("cd", 2)
        dictionary.add_ngram("da", 2)

        result = dictionary.tokenize_line("abcd")

        expected_source_seq = torch.tensor([[2, 3, 4, 5], [6, 8, 9, 10]])

        self.assertListEqual(result["source"].tolist(), expected_source_seq.tolist())

    def test_tokenizing_packed(self):
        dictionary = Dictionary(2, 10000, 10000, False, "dense", packed=True)

        dictionary.add_ngram("<unk>", 1)
        dictionary.add_ngram("<pad>", 1)
        dictionary.add_ngram("a", 1)
        dictionary.add_ngram("b", 1)
        dictionary.add_ngram("c", 1)
        dictionary.add_ngram("d", 1)

        dictionary.add_ngram("<unk>", 2)
        dictionary.add_ngram("<pad>", 2)
        dictionary.add_ngram("ab", 2)
        dictionary.add_ngram("bc", 2)
        dictionary.add_ngram("cd", 2)
        dictionary.add_ngram("da", 2)

        result = dictionary.tokenize_line("abcd")

        expected_source_seq = torch.tensor([393218, 524291, 589828, 655365])

        self.assertListEqual(result["source"].tolist(), expected_source_seq.tolist())

        expected_source_seq_unpack = [[2, 3, 4, 5], [6, 8, 9, 10]]
        self.assertListEqual(utils.unpack_tensor(result["source"]).tolist(), expected_source_seq_unpack)


    def test_same_embedding(self):

        torch.manual_seed(1234)

        emb = NGramsEmbedding(100, 100, packed=False)

        inp = torch.tensor([[1, 2, 3], [4, 5, 6]])

        # Sanity check
        self.assertTrue(torch.equal(inp.unsqueeze(-1), inp.unsqueeze(-1)))

        embedded_unpacked = emb(inp.unsqueeze(-1))
        self.assertTrue(torch.equal(embedded_unpacked, emb(inp.unsqueeze(-1))))

        packed_inp = utils.pack_tensor(inp).unsqueeze(-1)

        emb.packed = True

        embedded_packed = emb(packed_inp)

        self.assertTrue(torch.equal(embedded_unpacked, embedded_packed))

        


