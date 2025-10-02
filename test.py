# test.py
import argparse
import os
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode


def run_inference(
    image_path, model_path, out_dir="output_image", device="cpu", threshold=0.5
):
    os.makedirs(out_dir, exist_ok=True)

    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # for balloon dataset

    predictor = DefaultPredictor(cfg)

    img = cv2.imread(image_path)
    outputs = predictor(img)

    v = Visualizer(img[:, :, ::-1], scale=1.0, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result = out.get_image()[:, :, ::-1]

    save_path = os.path.join(out_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--model", default="./output/model_final.pth", help="Path to trained model"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "mps", "cuda"], help="Device"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold")
    args = parser.parse_args()

    run_inference(args.image, args.model, device=args.device, threshold=args.threshold)
