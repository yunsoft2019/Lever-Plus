import random

from .base_sampler import BaseSampler


class RandSampler(BaseSampler):
    def __init__(
        self,
        candidate_num,
        sampler_name,
        anchor_sample_num,
        index_ds_len,
        dataset_name,
        cache_dir,
        overwrite,
        anchor_idx_list=None,
        seed: int = 42,
    ):
        super().__init__(
            candidate_num=candidate_num,
            dataset_name=dataset_name,
            sampler_name=sampler_name,
            anchor_sample_num=anchor_sample_num,
            index_ds_len=index_ds_len,
            cache_dir=cache_dir,
            overwrite=overwrite,
            anchor_idx_list=anchor_idx_list,
        )
        self.seed = seed
        # 设置随机种子以确保可复现性
        random.seed(self.seed)

    def sample(self, anchor_set, train_ds):
        # 确保使用固定的随机种子
        random.seed(self.seed)
        candidate_set_idx = {}
        for s_idx in anchor_set:
            random_candidate_set = random.sample(
                range(0, len(train_ds)), self.candidate_num
            )
            while s_idx in random_candidate_set:
                random_candidate_set = random.sample(
                    list(range(0, len(train_ds))), self.candidate_num
                )
            candidate_set_idx[s_idx] = random_candidate_set
        return candidate_set_idx
