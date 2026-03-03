"""Plotting utilities without external plotting dependencies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


W, H = 900, 520
M = 60


def _save_canvas(draw_fn, outpath: str) -> None:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw_fn(draw)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    img.save(outpath)


def _scale(vals: np.ndarray, lo: float, hi: float, pix_lo: float, pix_hi: float) -> np.ndarray:
    denom = (hi - lo) if hi > lo else 1.0
    return pix_lo + (vals - lo) / denom * (pix_hi - pix_lo)


def _line_plot(x: np.ndarray, y_series: list[tuple[np.ndarray, str]], outpath: str, invert_y: bool = False) -> None:
    xlo, xhi = float(np.min(x)), float(np.max(x))
    all_y = np.concatenate([y for y, _ in y_series])
    ylo, yhi = float(np.min(all_y)), float(np.max(all_y))

    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle((M, M, W - M, H - M), outline="black", width=2)
        colors = ["blue", "red", "green", "purple", "orange"]
        for k, (y, _) in enumerate(y_series):
            xs = _scale(x, xlo, xhi, M + 10, W - M - 10)
            if invert_y:
                ys = _scale(y, ylo, yhi, M + 10, H - M - 10)
            else:
                ys = _scale(y, ylo, yhi, H - M - 10, M + 10)
            pts = [(float(px), float(py)) for px, py in zip(xs, ys)]
            draw.line(pts, fill=colors[k % len(colors)], width=3)

    _save_canvas(draw_fn, outpath)


def plot_audiogram(frequencies: np.ndarray, mu: np.ndarray, cov: np.ndarray, truth: np.ndarray | None, outpath: str) -> None:
    std = np.sqrt(np.diag(cov))
    lo = mu - 1.96 * std
    hi = mu + 1.96 * std
    x = np.log10(frequencies)
    series = [(mu, "mu"), (lo, "lo"), (hi, "hi")]
    if truth is not None:
        series.append((truth, "truth"))
    _line_plot(x, series, outpath, invert_y=True)


def plot_entropy_curve(entropy_curve: list[float], outpath: str) -> None:
    x = np.arange(len(entropy_curve), dtype=np.float64)
    y = np.asarray(entropy_curve, dtype=np.float64)
    _line_plot(x, [(y, "entropy")], outpath)


def plot_mae_curve(mae_curve: list[float], outpath: str) -> None:
    x = np.arange(len(mae_curve), dtype=np.float64)
    y = np.asarray(mae_curve, dtype=np.float64)
    _line_plot(x, [(y, "mae")], outpath)


def plot_variance_heatmap(cov_trace: list[np.ndarray], outpath: str) -> None:
    diag = np.vstack([np.diag(c) for c in cov_trace])
    norm = (diag - diag.min()) / (diag.max() - diag.min() + 1e-12)
    img = (norm * 255.0).astype(np.uint8)
    heat = Image.fromarray(img.T, mode="L").resize((W - 2 * M, H - 2 * M))
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(heat.convert("RGB"), (M, M))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(outpath)
