import torch
import torch.utils.benchmark as bm
from torch.cuda import device


from src.models.ngme import NGramsEmbedding


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

if __name__ == "__main__":
    timeit_unpacked("cpu")
    timeit_unpacked("cuda:0")

    timeit_packed("cpu")
    timeit_packed("cuda:0")


