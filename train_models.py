from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_setup import ROOT, enable_modules


def main():
    p = argparse.ArgumentParser(description="Train either main model with the archived reference recipe.")
    p.add_argument("--model", required=True, choices=("yolo11s", "ealc"))
    p.add_argument("--seed", type=int, choices=(3702, 42, 3407), default=3702)
    p.add_argument("--device", default="0")
    p.add_argument("--pretrained", type=Path, required=True, help="Path to externally obtained YOLO11s pretrained weights.")
    p.add_argument("--data", type=Path, default=ROOT / "test_only.yaml")
    p.add_argument("--out", type=Path, default=ROOT / "training_runs")
    args = p.parse_args()

    enable_modules()
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.nn.tasks_ealc_ms_intensity import EALCMSA
    from ultralytics.utils import RANK

    class EALCMSATrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = EALCMSA(cfg, nc=self.data["nc"], ch=self.data.get("channels", 3),
                           verbose=verbose and RANK == -1)
            if hasattr(self, "set_model_names_for_load"):
                model = self.set_model_names_for_load(model)
            if weights:
                model.load(weights)
            model.enable_ealc_v2()
            return model

    config = json.loads((ROOT / "configs" / "training_recipe.json").read_text(encoding="utf-8"))
    if not args.pretrained.is_file():
        raise FileNotFoundError(args.pretrained)
    if not args.data.is_file():
        raise FileNotFoundError("Prepare the reference dataset first, or pass --data.")
    config.update(data=str(args.data.resolve()), device=args.device, seed=args.seed,
                  project=str(args.out.resolve()), name=f"{args.model}_seed{args.seed}",
                  exist_ok=False, resume=False, save=True, plots=False, cache=False,
                  verbose=True)
    model = YOLO(str(args.pretrained.resolve()))
    if args.model == "ealc":
        model.train(trainer=EALCMSATrainer, **config)
    else:
        model.train(**config)


if __name__ == "__main__":
    main()

