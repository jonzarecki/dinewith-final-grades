from abc import abstractmethod

import numpy as np
import pytorch_lightning as pl
import sklearn
import tensorboard as tb
import tensorflow as tf
import torch
from common.constants import DATA_LONG_TERM_DIR
from common.experiments.config2args import make_config_loggable
from common.metrics.cluster_evaluation import silhouette_score
from common.visualizations.general import figure_to_image, plot_confusion_matrix
from torch import nn

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile  # fix tensorboard /tensorflow dependency issue with add_embeddings


class PLModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.all_hparams = hparams
        self.hparams = make_config_loggable(hparams)
        self.params_without_bn = []
        self.latest_orig_silhouette = -1
        # self.is_text = isinstance(self._get_dataset(True, True), torchtext.data.Dataset)

    def default_weight_initialization(self):
        self.params_without_bn = [
            params for name, params in self.named_parameters() if not ("_bn" in name or ".bn" in name)
        ]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def log_confusion_matrix(self, test_labels, test_pred):
        if self.logger is not None:
            # Calculate the confusion matrix.
            cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
            # Log the confusion matrix as an image summary.
            figure = plot_confusion_matrix(
                cm, class_names=[str(i) for i in range(len(np.unique(test_pred + test_labels)))]
            )
            cm_image = figure_to_image(figure)

            # Log the confusion matrix as an image summary.
            self.logger.experiment.add_image(
                "Confusion Matrix", torch.tensor(cm_image), self.global_step, dataformats="HWC"
            )

    def log_silhouette(self, embeddings, test_labels):
        self.latest_orig_silhouette = silhouette_score(embeddings, test_labels)
        return {f"all/silhouette_score": self.latest_orig_silhouette}

    def configure_optimizers(self):
        # REQUIRED
        optimizer = self.all_hparams.optimizer_lambda(self)
        scheduler = self.all_hparams.lr_scheduler_lambda(opt=optimizer, steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step" if self.hparams.lr_sched_type == "one-cycle" else "epoch"}
        ]

    def prepare_data(self):
        # download only
        self._get_dataset(False, True)
        self._get_dataset(True, True)

    def _get_dataset(self, train: bool, download: bool):
        ds = self.all_hparams.dataset(
            root=DATA_LONG_TERM_DIR, train=train, download=download, transform=self.all_hparams.transforms["train"]
        )
        self.classes = list(ds.classes)
        return ds

    @abstractmethod
    def _get_dataloader(self, train: bool, download: bool):
        pass

    def train_dataloader(self):
        # REQUIRED
        return self._get_dataloader(train=True, download=False)

    def val_dataloader(self):
        return self._get_dataloader(train=False, download=False)
