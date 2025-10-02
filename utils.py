import os
import json
import cv2

from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.config import get_cfg as _get_cfg
from detectron2 import model_zoo

# Optional: keep if using custom validation loss
from loss import ValidationLoss


def get_cfg(
    output_dir,
    learning_rate,
    batch_size,
    iterations,
    checkpoint_period,
    model,
    device,
    nmr_classes,
):
    cfg = _get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.VAL = ("val",)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.DEVICE = device
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = iterations
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nmr_classes
    cfg.OUTPUT_DIR = output_dir
    return cfg


def get_balloon_dicts(img_dir, json_file):
    """
    Parse VIA JSON annotations and create Detectron2 dataset dicts.
    """
    with open(json_file) as f:
        annotations = json.load(f)

    dataset_dicts = []

    for idx, v in enumerate(annotations.values()):
        record = {}
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []

        regions = v["regions"]
        if isinstance(regions, dict):
            regions = regions.values()

        for region in regions:
            shape_attrs = region["shape_attributes"]
            all_points_x = shape_attrs["all_points_x"]
            all_points_y = shape_attrs["all_points_y"]

            # Polygon for segmentation
            poly = []
            for x, y in zip(all_points_x, all_points_y):
                poly.extend([x, y])

            x0 = min(all_points_x)
            y0 = min(all_points_y)
            x1 = max(all_points_x)
            y1 = max(all_points_y)

            obj = {
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": [poly],  # <- required for Mask R-CNN
                "category_id": 0,
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_balloon_dataset(root_dir):
    """
    Register Balloon dataset for Detectron2.
    """
    for d in ["train", "val"]:
        img_dir = os.path.join(root_dir, d)
        json_file = os.path.join(img_dir, "via_region_data.json")
        DatasetCatalog.register(
            d, lambda d=d, j=json_file, i=img_dir: get_balloon_dicts(i, j)
        )
        MetadataCatalog.get(d).set(thing_classes=["balloon"])

    return 1  # only 1 class


def train(
    output_dir,
    data_dir,
    learning_rate,
    batch_size,
    iterations,
    checkpoint_period,
    device,
    model,
):
    nmr_classes = register_balloon_dataset(data_dir)
    cfg = get_cfg(
        output_dir,
        learning_rate,
        batch_size,
        iterations,
        checkpoint_period,
        model,
        device,
        nmr_classes,
    )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)

    # Optional: custom validation loss
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    trainer.resume_or_load(resume=False)
    trainer.train()
