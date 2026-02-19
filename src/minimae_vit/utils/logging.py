from __future__ import annotations

import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    def __init__(self, out_dir: Path, use_tb: bool = True) -> None:
        self.jsonl_path = out_dir / "metrics.jsonl"
        self.writer = SummaryWriter(str(out_dir / "tb")) if use_tb else None

    def log(self, step: int, payload: dict) -> None:
        line = {"step": step, **payload}
        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(line) + "\n")
        if self.writer:
            for k, v in payload.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, step)

    def close(self) -> None:
        if self.writer:
            self.writer.close()
