import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    def __init__(self, out_dir: str | Path, use_tb: bool = True, use_jsonl: bool = True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.use_jsonl = use_jsonl
        self.fp = open(self.out_dir / "metrics.jsonl", "a", encoding="utf-8") if use_jsonl else None
        self.tb = SummaryWriter(self.out_dir / "tb") if use_tb else None

    def log(self, step: int, metrics: dict):
        if self.fp:
            self.fp.write(json.dumps({"step": step, **metrics}) + "\n")
            self.fp.flush()
        if self.tb:
            for k, v in metrics.items():
                if isinstance(v, (float, int)):
                    self.tb.add_scalar(k, v, step)

    def close(self):
        if self.fp:
            self.fp.close()
        if self.tb:
            self.tb.close()
