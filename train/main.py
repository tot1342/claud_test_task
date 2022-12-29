from argparse import ArgumentParser
from train import FastRCNN, ObjectDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

torch.manual_seed(42)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--checkpoint_load_path", default="")
    parser.add_argument("--command", default="fit")
    parser.add_argument("--gpus", default=1)
    parser.add_argument("--max_epochs", default=1)
    parser.add_argument("--batch_size", default=3)
    parser.add_argument("--num_workers", default=3)
    parser.add_argument("--num_classes", default=91)

    parser.add_argument("--check_val_every_n_epoch", default=1)
    parser.add_argument("--default_root_dir", default="./")
    parser.add_argument("--logger_name", default="logs")
    parser.add_argument("--checkpoint_save_path", default="./W")
    parser.add_argument("--checkpoint_path", default="./")
    parser.add_argument("--save_top_k", default=1)
    parser.add_argument("--model_name", default="fasterrcnn")

    parser.add_argument("--image_train_path", default="./images/train2017/")
    parser.add_argument("--annotation_train_path", default="./annotations/instances_train2017.json")
    parser.add_argument("--image_val_path", default="./images/val2017/")
    parser.add_argument("--annotation_val_path", default="./annotations/instances_val2017.json")

    args = parser.parse_args()
    if args.checkpoint_load_path == "":
        model = FastRCNN(lr=args.learning_rate, num_classes=args.num_classes)
    else:
        model = FastRCNN.load_from_checkpoint(checkpoint_path=args.checkpoint_load_path)

    data_module = ObjectDataModule(image_train_path=args.image_train_path,
                                   image_val_path=args.image_val_path,
                                   annotation_train_path=args.annotation_train_path,
                                   annotation_val_path=args.annotation_val_path,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers
                                   )

    logger = TensorBoardLogger("tb_logs", name=args.logger_name)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.save_top_k,
        monitor="val_map",
        mode="max",
        dirpath=args.checkpoint_save_path,
        filename=args.model_name,
    )

    trainer = Trainer(gpus=args.gpus,
                      max_epochs=args.max_epochs,
                      check_val_every_n_epoch=args.check_val_every_n_epoch,
                      default_root_dir=args.default_root_dir,
                      logger=logger,
                      callbacks=checkpoint_callback
                      )

    if args.command == "fit":
        trainer.fit(model, data_module)
    elif args.command == "val":
        trainer.validate(model, data_module)
    elif args.command == "test":
        trainer.test(data_module)

