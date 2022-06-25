from typing import Callable

import numpy as np
import tensorboard as tb
import tensorflow as tf
import torch
from common.nn.lightning.pl_model import PLModel
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.utils.data import DataLoader

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile  # fix tensorboard /tensorflow dependency issue with add_embeddings


class ExampleEmbeddingModel(PLModel):
    # TODO: doesn't support multi-gpu ATM
    def __init__(
        self, base_model: nn.Module, head_model_lambda: Callable[[int], nn.Module], hparams, lbl_num=10, **kwargs
    ):
        super().__init__(hparams)

        self.lbl_num = lbl_num
        self.model = base_model
        self.loss = nn.CrossEntropyLoss()

        self.head = head_model_lambda(self.lbl_num)
        self.silhouette_log_freq = 1

        self.default_weight_initialization()

    def forward(self, x, return_emb=False):
        embedding = self.model(x)
        output = self.head(embedding)
        return (embedding, output) if return_emb else output

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x, return_emb=False)

        loss = self.loss(y_hat, y)

        tensorboard_logs = {"all/train_loss": loss, "all/lr": self.trainer.lr_schedulers[0]["scheduler"].get_lr()[0]}
        return {"loss": loss, "log": tensorboard_logs}

    def _get_acc(self, y_pred, y_true):
        # calc task acc
        rel_idxs = y_true != -1
        y_true, y_pred = y_true[rel_idxs], y_pred[rel_idxs]
        max_vals, max_indices = torch.max(y_pred, 1)
        # train_acc = (max_indices == Y).sum().data.numpy() / max_indices.size()[0]
        acc = (max_indices == y_true).sum().item() / max_indices.size()[0]
        return acc

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        embedding, y_hat = self.forward(x, return_emb=True)
        loss = self.loss(y_hat, y)

        return {
            "loss": loss,
            "accuracy": accuracy(y_hat, y, num_classes=self.lbl_num),
            "embedding": embedding.cpu(),
            "y_pred": y_hat.cpu(),
            "y_true": y.cpu(),
        }

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        tqdm_log = {}
        tensorboard_logs = {}
        check_trigger = lambda log_freq: self.current_epoch % max(1, log_freq - 1) == 0 and self.current_epoch != 0
        log_conf = check_trigger(self.confusion_matrix_log_freq)
        log_embeddings = check_trigger(self.embeddings_log_freq)
        log_silhouette = check_trigger(self.silhouette_log_freq)

        test_pred_raw = torch.cat([out["y_pred"] for out in outputs])
        embeddings = torch.cat([out["embedding"] for out in outputs])
        test_labels = torch.cat([out["y_true"] for out in outputs]).tolist()
        test_pred = list(np.argmax(test_pred_raw, axis=1))
        if log_conf:
            self.log_confusion_matrix(test_labels, test_pred)
        if log_silhouette:
            tensorboard_logs.update(self.log_silhouette(embeddings, test_labels))
        if log_embeddings:
            self.logger.experiment.add_embedding(
                embeddings[:1000],
                metadata=[self.classes[n] for n in test_labels[:1000]],
                # label_img=X_test_dict['img'].data,  # don't save image, not interesting
                global_step=self.global_step,
            )
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        tqdm_log["all/val_acc"] = np.mean([x["accuracy"] for x in outputs])
        tensorboard_logs.update(tqdm_log)
        tensorboard_logs["all/val_loss"] = avg_loss
        return {"val_loss": avg_loss, "progress_bar": tqdm_log, "log": tensorboard_logs}

    def _get_dataloader(self, train: bool, download: bool):
        dataset = self._get_dataset(train, download)
        assert self.all_hparams.train_size is None, "train size is not supported"

        loader = DataLoader(
            dataset,
            batch_size=self.all_hparams.batch_size,
            drop_last=True,
            shuffle=train,
            **({"num_workers": 6, "pin_memory": True} if self.on_gpu else {})
        )
        return loader
