import argparse
from utils import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Detectron2 Balloon Detector")

    parser.add_argument(
        "--data-dir", default="./data", help="Path to dataset root directory"
    )
    parser.add_argument(
        "--output-dir", default="./output", help="Directory to save outputs"
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to train on: cpu or gpu"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.00025, help="Learning rate"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--iterations", type=int, default=3000, help="Number of training iterations"
    )
    parser.add_argument(
        "--checkpoint-period", type=int, default=500, help="Checkpoint period"
    )
    parser.add_argument(
        "--model",
        default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        help="Model from Detectron2 model zoo",
    )

    args = parser.parse_args()

    train(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        iterations=args.iterations,
        checkpoint_period=args.checkpoint_period,
        device=args.device,
        model=args.model,
    )
