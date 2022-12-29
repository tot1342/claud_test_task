import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pytorch_lightning import LightningDataModule, LightningModule
import numpy as np
import os
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from pycocotools.coco import COCO

torch.manual_seed(42)


class DatasetCOCO(torch.utils.data.Dataset):
    """
        Кастомный датасет для подгрузки данных в анотации COCO
    """

    def __init__(self, root, annotation, transform=None, tr=False):
        self.root = root
        self.transforms = transform
        self.coco = COCO(annotation)
        if tr:
            self.ids = list(sorted(self.coco.imgs.keys()))
        else:
            self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path))
        num_objs = len(coco_annotation)

        # иногда на изображении нет объектов
        k = 1
        while num_objs == 0:
            img_id = self.ids[index + k]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            coco_annotation = coco.loadAnns(ann_ids)
            path = coco.loadImgs(img_id)[0]['file_name']
            img = Image.open(os.path.join(self.root, path))
            num_objs = len(coco_annotation)
            k += 1

        img = np.array(img)
        img[img > 255] = 255
        img[img < 0] = 0
        if len(img.shape) == 2:
            img = np.concatenate([img[:, :, np.newaxis], img[:, :, np.newaxis], img[:, :, np.newaxis]], axis=0)
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)

        if self.transforms is not None:
            img = self.transforms(img)

        boxes = []
        for i in range(num_objs):
            xmin, ymin = coco_annotation[i]['bbox'][0], coco_annotation[i]['bbox'][1]
            xmax, ymax = xmin + coco_annotation[i]['bbox'][2], ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = []
        for i in range(num_objs):
            labels.append(coco_annotation[i]["category_id"])
        labels = torch.tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([img_id])

        my_annotation = {"boxes": boxes, "labels": labels}

        return img.float(), my_annotation

    def __len__(self):
        return len(self.ids)


class ObjectDataModule(LightningDataModule):
    """
        Класс для создания даталоудеров
    """

    def __init__(
            self,
            image_train_path,
            image_val_path,
            annotation_train_path,
            annotation_val_path,
            batch_size,
            num_workers
    ):
        super().__init__()
        self.image_train_path = image_train_path
        self.image_val_path = image_val_path
        self.annotation_train_path = annotation_train_path
        self.annotation_val_path = annotation_val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        self.coco_train = DatasetCOCO(root=self.image_train_path,
                                      annotation=self.annotation_train_path,
                                      transform=self.transform,
                                      tr=True
                                      )

        self.coco_val = DatasetCOCO(root=self.image_val_path,
                                    annotation=self.annotation_val_path,
                                    transform=self.transform
                                    )

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def train_dataloader(self):
        train_dataloader = DataLoader(self.coco_train,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      collate_fn=self.collate_fn
                                      )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.coco_val,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    collate_fn=self.collate_fn
                                    )
        return val_dataloader


class FastRCNN(LightningModule):
    """
        Модель в анотации torch lightning
    """

    def __init__(self, lr, num_classes):
        super().__init__()
        self.lr = lr
        self.metric = MeanAveragePrecision()
        self.num_classes = num_classes
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

    def forward(self, z):
        return self.model(z)

    def training_step(self, batch, batch_idx):
        imgs, target = batch
        loss = self.model(imgs, target)
        loss_all = loss['loss_classifier'] + loss['loss_box_reg'] + loss['loss_objectness'] + loss['loss_rpn_box_reg']
        self.log("loss_classifier", loss['loss_classifier'])
        self.log("loss_box_reg", loss['loss_box_reg'])
        self.log("loss_objectness", loss['loss_objectness'])
        self.log("loss_rpn_box_reg", loss['loss_rpn_box_reg'])
        self.log("loss", loss_all)
        return loss_all

    def validation_step(self, batch, batch_idx):
        imgs, target = batch
        with torch.no_grad():
            out = self.model(imgs)
        self.metric.update(out, target)
        metrics = self.metric.compute()
        self.log("val_map", torch.Tensor([metrics["map"]]))
        self.log("val_map_50", torch.Tensor([metrics["map_50"]]))
        self.log("val_map_75", torch.Tensor([metrics["map_75"]]))

        return metrics["map"]

    def configure_optimizers(self):
        opt_g = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return opt_g

