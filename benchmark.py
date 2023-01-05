import torch
import torch.utils.benchmark as bm


from src.models.ngme import NGramsEmbedding


if __name__ == "__main__":

    input_packed = torch.randint(0, 60000, (100, 10))
    input_unpacked = torch.randint(0, 60000, (4, 100, 10))

    embedding = NGramsEmbedding(60000, 512, packed=False)

    t0 = bm.Timer(
        stmt="embedding(input_unpacked)",
        globals = {"input_unpacked": input_unpacked, "embedding": embedding}
    )

    print(t0.timeit())


