import time
import timeit

import numpy as np
import torch
import torch.utils.benchmark as bm
from torch.cuda import device

from prettytable import PrettyTable

from src.models.ngme import NGramsEmbedding, NGramsEmbeddingFast
from src.utils import concat_dataset, numba_concat_dataset


def timeit_unpacked(device_ctx: str):

    with device(device_ctx):
        input_unpacked = torch.randint(0, 100, (4, 10, 10))

        embedding = NGramsEmbedding(100, 64, packed=False)

        t0 = bm.Timer(
            stmt="embedding(input_unpacked)",
            globals = {"input_unpacked": input_unpacked, "embedding": embedding}
        )

        print("Unpacked")
        print(t0.timeit())

def timeit_packed(device_ctx: str):
    with device(device_ctx):
        input_packed = torch.randint(0, 100, (10, 10))

        embedding = NGramsEmbedding(100, 64, packed=True)

        t0 = bm.Timer(
            stmt="embedding(input_unpacked)",
            globals = {"input_packed": input_packed, "embedding": embedding}
        )
        
        print("Packed")
        print(t0.timeit())

def timeit_embed(device="cpu"):

    # ========================================================================================
    input_unpacked = torch.randint(1, 100, (4, 10, 4)).to(device)
    embedding_py_unpacked = NGramsEmbedding(100, 64, packed=False).to(device)

    t0 = bm.Timer(
        stmt="embedding_py_unpacked(input_unpacked)",
        globals = {"input_unpacked": input_unpacked, "embedding_py_unpacked": embedding_py_unpacked}
    )
    time_py_unpacked = t0.timeit()

    # ========================================================================================
    input_packed = torch.randint(1, 100, (10, 10)).to(device)
    embedding_py_packed = NGramsEmbedding(100, 64, packed=True).to(device)

    t0 = bm.Timer(
        stmt="embedding_py_packed(input_packed)",
        globals = {"input_packed": input_packed, "embedding_py_packed": embedding_py_packed}
    )
    time_py_packed = t0.timeit()

    # ========================================================================================
    input_unpacked = torch.randint(1, 100, (4, 10, 4), dtype=torch.int64).to(device)
    embedding_cpp_unpacked = NGramsEmbeddingFast(100, 64, False).to(device)

    t0 = bm.Timer(
        stmt="embedding_cpp_unpacked(input_unpacked)",
        globals = {"input_unpacked": input_unpacked, "embedding_cpp_unpacked": embedding_cpp_unpacked}
    )
    time_cpp_unpacked = t0.timeit()
    
    # ========================================================================================
    input_packed = torch.randint(1, 100, (10, 10), dtype=torch.int64).to(device)
    embedding_cpp_packed = NGramsEmbeddingFast(100, 64, True).to(device)

    t0 = bm.Timer(
        stmt="embedding_cpp_packed(input_packed)",
        globals = {"input_packed": input_packed, "embedding_cpp_packed": embedding_cpp_packed}
    )
    time_cpp_packed = t0.timeit()

    table = PrettyTable(["Function", "Implementation", "Time (sec)"])
    table.add_row(["Embedding Unpacked", "Python", round(time_py_unpacked.raw_times[0], 2)])
    table.add_row(["Embedding Packed", "Python", round(time_py_packed.raw_times[0], 2)])
    table.add_row(["Embedding Unpacked", "C++", round(time_cpp_unpacked.raw_times[0], 2)])
    table.add_row(["Embedding Packed", "C++", round(time_cpp_packed.raw_times[0], 2)])

    print(table)

def timeit_pack_cpp():
    # Pack/Unpack function with Python
    time_pack_py = timeit.timeit("pack([1, 2, 3, 4])", setup="from src.utils import pack", number=1_000_000)
    time_unpack_py = timeit.timeit("unpack(1125912791875585)", setup="from src.utils import unpack", number=1_000_000)

    # Pack/Unpack function with C++
    time_pack_cpp = timeit.timeit("pack([1, 2, 3, 4])", setup="from utils_cpp import pack", number=1_000_000)
    time_unpack_cpp = timeit.timeit("unpack(1125912791875585)", setup="from utils_cpp import unpack", number=1_000_000)

    time_unpack_as_tensor_cpp = timeit.timeit("unpack_as_tensor(1125912791875585)", setup="from utils_cuda import unpack_as_tensor", number=1_000_000)

    time_pack_as_tensor_cpp = timeit.timeit("pack([1, 2, 3, 4])", setup="from utils_cuda import pack", number=1_000_000)


    table = PrettyTable(["Function", "Implementation", "Time (sec)"])
    table.add_row(["pack", "Python", round(time_pack_py, 2)])
    table.add_row(["unpack", "Python", round(time_unpack_py, 2)])

    table.add_row(["pack", "C++", round(time_pack_cpp, 2)])
    table.add_row(["unpack", "C++", round(time_unpack_cpp, 2)])

    table.add_row(["pack (fast)", "C++", round(time_pack_as_tensor_cpp, 2)])
    table.add_row(["unpack (fast)", "C++", round(time_unpack_as_tensor_cpp, 2)])
    print(table)


def timeit_n_hot():
    time_n_hot_py = timeit.timeit("n_hot(torch.randint(0, 10, (4, 10, 10)), 10, False)", setup="import torch; from src.models.ngme import n_hot", number=1_000_00)

    time_n_hot_cpp = timeit.timeit("n_hot(torch.randint(0, 10, (4, 10, 10)), 10, False)", setup="import torch; from ngme_cpp import n_hot", number=1_000_00)

    table = PrettyTable(["Function", "Implementation", "Time (sec)"])

    table.add_row(["n_hot", "Python", round(time_n_hot_py, 2)])
    table.add_row(["n_hot", "C++", round(time_n_hot_cpp, 2)])

    print(table)


def timeit_n_hot_packed():

    exec_code = """n_hot(pack_tensor(torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]).t().contiguous()).unsqueeze(-1), 87, True)"""
    time_n_hot_py = timeit.timeit(exec_code, setup="import torch; from src.models.ngme import n_hot; from src.utils import pack_tensor", number=1_000_00)

    time_n_hot_cpp = timeit.timeit(exec_code, setup="import torch; from ngme_cpp import n_hot, pack_tensor;", number=1_000_00)

    table = PrettyTable(["Function", "Implementation", "Time (sec)"])

    table.add_row(["n_hot", "Python", round(time_n_hot_py, 2)])
    table.add_row(["n_hot", "C++", round(time_n_hot_cpp, 2)])

    print(table)

def timeit_pack_tensor():
    time_pack_tensor_py = timeit.timeit("pack_tensor(torch.ones((4, 10), dtype=torch.int64))", setup="import torch; from src.utils import pack_tensor", number=1_000_00)

    time_pack_tensor_cpp = timeit.timeit("pack_tensor(torch.ones((4, 10), dtype=torch.int64))", setup="import torch; from ngme_cpp import pack_tensor", number=1_000_00)
    table = PrettyTable(["Function", "Implementation", "Time (sec)"])

    table.add_row(["pack_tensor", "Python", round(time_pack_tensor_py, 2)])
    table.add_row(["pack_tensor", "C++", round(time_pack_tensor_cpp, 2)])

    print(table)

def timeit_unpack_tensor():
    time_pack_tensor_py = timeit.timeit("unpack_tensor(torch.tensor([327681, 393218, 458755, 524292]))", setup="import torch; from src.utils import unpack_tensor", number=1_000_00)

    time_pack_tensor_cpp = timeit.timeit("unpack_tensor(torch.tensor([327681, 393218, 458755, 524292]))", setup="import torch; from ngme_cpp import unpack_tensor", number=1_000_00)
    table = PrettyTable(["Function", "Implementation", "Time (sec)"])

    table.add_row(["unpack_tensor", "Python", round(time_pack_tensor_py, 2)])
    table.add_row(["unpack_tensor", "C++", round(time_pack_tensor_cpp, 2)])

    print(table)


def timeit_np_concat():

    array = np.ones((1000, 2, 400), dtype=np.int16).tolist()

    start = time.time()
    concat_dataset(array)
    end = time.time()
    print("Elapsed = %s" % (end - start))

    array = np.ones((1000, 2, 400)).tolist()
    numba_concat_dataset(array)
    start = time.time()
    numba_concat_dataset(array)
    end = time.time()
    print("Elapsed  = %s" % (end - start))

    # table = PrettyTable(["Function", "Implementation", "Time (sec)"])
    # table.add_row(["concat", "Python", round(time_concat, 2)])
    # table.add_row(["concat(jit)", "Python", round(time_concat_jit, 2)])

    # print(table)

if __name__ == "__main__":
    timeit_pack_cpp()
    timeit_n_hot()