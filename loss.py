# loss.py
from detectron2.engine import HookBase
from detectron2.data import build_detection_test_loader
import detectron2.utils.comm as comm
import torch


class ValidationLoss(HookBase):
    """
    A training hook that computes validation loss periodically and logs it in the trainer storage.

    Usage:
        hook = ValidationLoss(cfg, eval_period=500)
        trainer.register_hooks([hook])

    Args:
        cfg: Detectron2 CfgNode used to build the validation dataloader. cfg.DATASETS.VAL must be set.
        eval_period (int): run validation loss every `eval_period` iterations. If 1, runs every iteration.
    """

    def __init__(self, cfg, eval_period: int = 500):
        super().__init__()
        self.cfg = cfg.clone()
        # make sure VAL dataset is configured
        val_datasets = getattr(self.cfg.DATASETS, "VAL", None)
        if not val_datasets or len(val_datasets) == 0:
            raise ValueError(
                "cfg.DATASETS.VAL must be set to a validation dataset name (e.g. ('val',))"
            )
        self.val_dataset_name = val_datasets[0]
        self.eval_period = int(eval_period)

        # build an iterator for the validation loader
        self._loader = iter(
            build_detection_test_loader(self.cfg, self.val_dataset_name)
        )

    def _get_next_batch(self):
        """
        Get next batch from validation loader; if exhausted, recreate iterator and return first batch.
        """
        try:
            batch = next(self._loader)
        except StopIteration:
            # rebuild iterator and get first batch
            self._loader = iter(
                build_detection_test_loader(self.cfg, self.val_dataset_name)
            )
            batch = next(self._loader)
        return batch

    def after_step(self):
        """
        Called after each training step. Computes validation loss every `eval_period` training iterations.
        """
        # trainer.storage.iter or self.trainer.iter gives current iteration (0-based). Use +1 to be 1-based.
        current_iter = self.trainer.iter + 1
        if current_iter % self.eval_period != 0:
            return

        # get a batch from validation loader
        data = self._get_next_batch()

        # compute loss in eval mode and without grad
        with torch.no_grad():
            model = self.trainer.model
            was_training = model.training
            try:
                model.eval()
                loss_dict = model(data)  # expects a dict of loss tensors
            finally:
                # restore training mode
                if was_training:
                    model.train()

        # validate loss is finite
        losses = sum(loss_dict.values())
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                f"Validation losses contain non-finite values: {loss_dict}"
            )

        # reduce across all workers
        # comm.reduce_dict expects a dict mapping string -> tensor
        loss_dict_reduced = {
            ("val_" + k): v.item() for k, v in comm.reduce_dict(loss_dict).items()
        }

        total_val_loss = sum(v for v in loss_dict_reduced.values())

        # log to storage on main process
        if comm.is_main_process():
            # put_scalars expects keyword args of scalar values
            self.trainer.storage.put_scalars(
                total_val_loss=total_val_loss, **loss_dict_reduced
            )
