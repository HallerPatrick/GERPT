from src.dataset import GenericDataModule, ShardedDataModule, SplitDataModule
from src.processor import Processor


def get_split(sample, split):
    """Get the split from the sample"""
    return sample[split]


class TrainStrategy:
    """Train the model with the given data module"""

    @staticmethod
    def train_from_strategy(trainer, model, args):
        """Train the model with the given data module"""

        write_strategy = Processor.get_strategy_from_path(args.saved_data)

        if write_strategy == "default":
            TrainStrategy.train_default(trainer, model, write_strategy, args)
        elif write_strategy == "split":
            TrainStrategy.train_split(trainer, model, args)
        elif write_strategy == "sharded":
            TrainStrategy.train_sharded(trainer, model, write_strategy, args)
        else:
            raise ValueError(f"Unknown write strategy: {write_strategy}")

    @staticmethod
    def train_default(trainer, model, write_strategy, args):
        """Train the model with the default data module"""
        tokenized_dataset = Processor.from_strategy(write_strategy).read_dataset(
            args.saved_data
        )
        data_module = GenericDataModule(
            tokenized_dataset, args.batch_size, args.bptt, None, args.cpus
        )

        # TRAIN!
        trainer.fit(model, data_module)

    @staticmethod
    def train_split(trainer, model, args):
        """Train the model with the split data module. The module is a wrapper around
        the DatasetIterator class, that yields splits of the dataset.
        """
        data_module = SplitDataModule(
            args.saved_data, args.batch_size, args.bptt, None, args.cpus
        )
        # TRAIN!
        trainer.fit(model, data_module)
        trainer.fit_loop.epoch_progress.reset_on_run()

    @staticmethod
    def train_sharded(trainer, model, write_strategy, args):
        """Train the model with the sharded data module"""
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")

            tokenized_dataset = Processor.from_strategy(write_strategy).read_dataset(
                args.saved_data
            )
            for idx, shard in enumerate(tokenized_dataset):
                print(f"Shard {idx + 1}")

                trainer.fit_loop.epoch_progress.reset_on_run()

                if idx == args.epochs:
                    break

                train = get_split(shard, "train")
                test = get_split(shard, "test")
                valid = get_split(shard, "valid")
                data_module = SplitDataModule(
                    train, test, valid, args.batch_size, args.bptt, None, args.cpus
                )

                trainer.fit(model, data_module)
