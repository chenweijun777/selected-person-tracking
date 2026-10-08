from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter


class StageTimer:
    def __init__(self) -> None:
        self.samples_ms: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.samples_ms[stage].append((perf_counter() - started) * 1000.0)

    def summary(self) -> dict[str, dict[str, float]]:
        import numpy as np

        return {
            stage: {
                "count": float(len(values)),
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "mean_ms": float(np.mean(values)),
            }
            for stage, values in self.samples_ms.items()
            if values
        }
