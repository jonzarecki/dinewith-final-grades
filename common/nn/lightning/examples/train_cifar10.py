import argparse
import copy
import time

import pytorch_lightning as pl
from common.constants import TENSORBOARD_DIR
from common.experiments.config2args import convert_config_to_args
from common.experiments.file_util import using_debugger
from common.experiments.offline_run import run_offline_if_requested
from common.nn.lightning.examples.cifar10_module import CIFAR10_Module
from common.nn.lightning.examples.config_helper import add_to_config
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


def define_argparser():
    parser = argparse.ArgumentParser(description="Boostrapped-Clustering-AL")
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=[
            "mnist",
            "fashion-mnist",
            "cifar10",
            "cifar100",
            "mini-imagenet",
            "omniglot",
            "kmnist",
            "3class-mnist",
            "stanford-cars",
            "trec",
        ],
        default="cifar10",
        help="name of dataset to train on (default: cifar10)",
    )
    parser.add_argument(
        "--train-noise", type=float, default=0.0, help="amount of label noise to add to the training set"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        choices=[
            "simple-cnn",
            "mlp",
            "resnet18",
            "vgg11",
            "vgg8b",
            "bert-gru",
            "resnet101",
            "resnet50",
            "densenet-bc-169",
            "wresnet28_10",
            "wresnet40_2",
        ],
        default="resnet18",
        help="backbone model (default: mlp)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=256, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=50, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument(
        "--loss",
        type=str,
        choices=["ce", "partial-ce", "ce-ls", "bge-logit", "scaled"],
        default="ce",
        help="loss function",
    )
    parser.add_argument("--smooth-capacity", type=float, default=0.1, help="capacity to smooth, smaller than 1")

    # optimizer
    parser.add_argument(
        "--optimizer-name", type=str, choices=["adam", "sgd", "adamw", "adagrad", "adadelta", "rmsprop"], default="sgd"
    )
    # lr scheduling
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR", help="learning rate (default: 1e-3)")
    parser.add_argument(
        "--lr-sched-type",
        type=str,
        choices=["multi-step", "constant", "cyclic", "cosine", "one-cycle"],
        default="constant",
        help="lr scheduler type",
    )
    parser.add_argument("--lr-sched-gamma", type=float, required=False, help="gamma param for multi-step lr-sched")
    parser.add_argument("--lr-sched-step-size", type=float, required=False, help="step up size for cyclic lr-sched")
    parser.add_argument(
        "--lr-sched-milestones", nargs="+", type=int, required=False, help="milestones for multi-step lr-sched"
    )

    # other
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--gpus", nargs="+", type=int, required=False, default=None, help="gpu ids to use for training")
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        metavar="S",
        help="amount of examples in train (default: no restriction)",
    )
    parser.add_argument(
        "--test-size", type=int, default=None, metavar="S", help="amount of examples in test (default: no restriction)"
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    parser.add_argument("--log-eval-freq", type=int, default=2, help="Epoch freq for evaluatiing")
    parser.add_argument("--offline", action="store_true", default=False, help="Runs the model offline using nohup")

    parser.add_argument("--expr-name", type=str, default="", help='Experiment name (default: "")')

    return parser


def main():
    convert_config_to_args()
    run_offline_if_requested()

    # print("waiting 2 hours")
    # time.sleep(2*60*60)

    # Training settings
    parser = define_argparser()
    args = parser.parse_args()
    # save_all_py_files(args.expr_name)
    orig_args = copy.copy(args)
    print(args)

    pl.seed_everything(args.seed)
    args = add_to_config(args)

    key = lambda s: f"orig/{args.dataset_name}/{s}/accuracy"
    results = {key("train"): [], key("test"): []}
    st = time.time()
    print(f"Calculating on {args.dataset_name} with {args.train_size} examples")
    base_model = args.model_lambda()

    model = CIFAR10_Module(base_model, args.head_lambda, args)

    logger = TensorBoardLogger(
        TENSORBOARD_DIR,
        name=f"{args.expr_name}/{args.dataset_name}/{args.model_name}_{args.optimizer_name}/"
        f"{args.train_size}_ex_{args.epochs}_eps",
    )
    args.use_gpu = args.gpus is not None and not args.no_cuda and not using_debugger
    trainer = Trainer(
        gpus=args.gpus if args.use_gpu else None,  # , distributed_backend='dp',
        check_val_every_n_epoch=args.log_eval_freq,
        logger=logger if not using_debugger else False,
        benchmark=True,
        max_epochs=args.epochs,
        min_epochs=args.epochs,
        num_sanity_val_steps=2,
        progress_bar_refresh_rate=2,
        precision=32,
    )  # 16 if args.use_gpu else 32)  # no speed boost ATM, only mem boost
    trainer.fit(model)

    # latest_val_acc = trainer.callback_metrics['all/orig_val_acc']
    print(trainer.callback_metrics)

    print(f"Took {int((time.time() - st) / 60)} minutes")

    print(results)

    # tensorboard serve --logdir /media/yonatanz/yz/boostrapped_clustering_al/logs --window_title boostrapped_clustering_al --max_reload_threads 5 --host 0.0.0.0 --port 8190

    # run tensorboard in ctx20
    # tensorboard serve --logdir /cortex/users/jonzarecki/long_term/boostrapped_clustering_al/logs --window_title boostrapped_clustering_al --max_reload_threads 10 --host 0.0.0.0 --port 8190 &
    # ctx20-up
    # tb-ctx20-up


if __name__ == "__main__":
    main()
