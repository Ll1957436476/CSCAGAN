# data/__init__.py
import importlib
import torch
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.base_dataset import BaseDataset


# ---- 顶层 worker_init_fn：Windows 多进程可被正确 pickling ----
def worker_init_fn(worker_id):
    import random, numpy as np, torch
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed ^ worker_id)
    random.seed(seed ^ worker_id)


def find_dataset_using_name(dataset_name: str):
    """
    根据 --dataset_mode 寻找并返回数据集类（必须继承 BaseDataset）。
    例如 dataset_name="aligned" -> 导入 data/aligned_dataset.py 中的 AlignedDataset。
    """
    dataset_filename = f"data.{dataset_name}_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"  # 大小写不敏感
    for name, cls in datasetlib.__dict__.items():
        # 只在对象是“类”时再判断是否继承 BaseDataset，避免 TypeError
        if isinstance(cls, type) and name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls
            break

    if dataset is None:
        # 用异常而非 print+exit，便于日志与上层捕获
        raise ImportError(
            f"In {dataset_filename}.py, expected a subclass of BaseDataset named {target_dataset_name} (case-insensitive)."
        )
    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    dataset_class = find_dataset_using_name(opt.dataset_mode)
    instance = dataset_class()
    instance.initialize(opt)
    print("dataset [%s] was created" % (instance.name()))
    return instance


def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader


# ---- 封装的 DataLoader：开启 pin_memory / 可选持久 worker / 预取 ----
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return "CustomDatasetDataLoader"

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = create_dataset(opt)

        num_workers = int(getattr(opt, "num_threads", 0))
        loader_kwargs = dict(
            batch_size=opt.batch_size,
            shuffle=not getattr(opt, "serial_batches", False),
            num_workers=num_workers,
            pin_memory=True,      # 配合张量 .to(device, non_blocking=True) 实现异步搬运
            drop_last=False,      # IN 下保持 False；如改 BN 可设 True
        )

        # 仅在多进程时启用持久 worker / 预取 / worker_init_fn
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2  # 需要 PyTorch>=1.7
            loader_kwargs["worker_init_fn"] = worker_init_fn

        # 兼容老 PyTorch（不支持 prefetch_factor/persistent_workers）自动降级重试
        try:
            self.dataloader = torch.utils.data.DataLoader(self.dataset, **loader_kwargs)
        except TypeError:
            loader_kwargs.pop("prefetch_factor", None)
            loader_kwargs.pop("persistent_workers", None)
            loader_kwargs.pop("worker_init_fn", None)
            self.dataloader = torch.utils.data.DataLoader(self.dataset, **loader_kwargs)

    def load_data(self):
        return self

    # def __len__(self):
    #     return min(len(self.dataset), int(getattr(self.opt, "max_dataset_size", float("inf"))))

    def __len__(self):
        # min 会把 inf “吃掉”，结果是 int 或可安全转为 int
        return int(min(len(self.dataset), getattr(self.opt, "max_dataset_size", float("inf"))))

    # def __iter__(self):
    #     max_sz = int(getattr(self.opt, "max_dataset_size", float("inf")))
    #     for i, data in enumerate(self.dataloader):
    #         if i * self.opt.batch_size >= max_sz:
    #             break
    #         yield data

    def __iter__(self):
        # 计算一个“有效的上限”，可能是数据集长度，也可能是用户给的有限数字
        effective_max = min(len(self.dataset), getattr(self.opt, "max_dataset_size", float("inf")))
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= effective_max:
                break
            yield data
