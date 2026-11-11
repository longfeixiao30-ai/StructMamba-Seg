from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class RareProbSchedulerHook(Hook):
    """
    按训练阶段动态调整 RareClassRandomCrop.rare_prob
    默认策略：
      0    ~ 7999   -> 0.15
      8000 ~ 39999  -> 0.50
      40000~ end    -> 0.30
    """

    def __init__(self, milestones=(0, 8000, 40000), probs=(0.15, 0.50, 0.30)):
        assert len(milestones) == 3
        assert len(probs) == 3
        self.milestones = milestones
        self.probs = probs
        self._last_prob = None

    def _find_rare_crop(self, runner):
        dataset = runner.train_dataloader.dataset

        # 防止外面再包一层 dataset wrapper
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        pipeline = getattr(dataset, 'pipeline', None)
        transforms = getattr(pipeline, 'transforms', None)
        if transforms is None:
            return None

        for t in transforms:
            if t.__class__.__name__ == 'RareClassRandomCrop':
                return t
        return None

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        cur_iter = runner.iter

        if cur_iter < self.milestones[1]:
            prob = self.probs[0]
        elif cur_iter < self.milestones[2]:
            prob = self.probs[1]
        else:
            prob = self.probs[2]

        if prob != self._last_prob:
            t = self._find_rare_crop(runner)
            if t is not None:
                t.rare_prob = prob
                runner.logger.info(
                    f'[RareProbSchedulerHook] Set RareClassRandomCrop.rare_prob = {prob:.2f} at iter {cur_iter}'
                )
            self._last_prob = prob