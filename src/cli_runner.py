import argparse, json, os, numpy as np, torch
from .config import Config
from .data import build_dataframe, make_loaders
from .model import create_model, load_checkpoint
from .eval import infer_logits, evaluate_multilabel, sigmoid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--primary", default="Pneumonia")
    ap.add_argument("--targets", nargs="+", default=["Pneumonia","Effusion","Atelectasis","Cardiomegaly"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        csv_path=args.csv,
        checkpoint_path=args.checkpoint,
        target_classes=args.targets,
        primary_label=args.primary,
    )
    cfg.fix_seeds()

    df = build_dataframe(cfg.data_dir, cfg.csv_path, cfg.target_classes)
    _, test_loader = make_loaders(df, cfg.target_classes, cfg.img_size, cfg.batch_size, cfg.num_workers, cfg.mean, cfg.std)

    model = create_model(num_classes=len(cfg.target_classes), pretrained=False, device=cfg.device)
    model = load_checkpoint(model, cfg.checkpoint_path, cfg.device)
    model.to(cfg.device)

    logits, labels, meta = infer_logits(model, test_loader, device=cfg.device)
    metrics = evaluate_multilabel(logits, labels, cfg.target_classes)

    # Simple corruption sweep (gaussian noise severity 1..3)
    from torchvision import transforms
    from PIL import Image
    from .data import CXR14Dataset
    # We'll just re-use the test loader images as-is; for brevity, corruption sweeps are omitted in CLI to keep runtime reasonable.

    out = {
        "targets": cfg.target_classes,
        "metrics_clean": metrics,
        "n_test": int(labels.shape[0]),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
