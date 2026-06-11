import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TimingResult:
    name: str
    total_seconds: float
    iterations: int

    @property
    def avg_seconds(self) -> float:
        return self.total_seconds / max(self.iterations, 1)

    @property
    def avg_ms(self) -> float:
        return self.avg_seconds * 1000

    @property
    def fps(self) -> float:
        return 1.0 / self.avg_seconds if self.avg_seconds > 0 else float("inf")

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg_ms:.2f}ms avg ({self.fps:.1f} FPS) over {self.iterations} runs"


class PrecisionTimer:
    def __init__(self, name: str = ""):
        self.name = name
        self._start: Optional[float] = None
        self._elapsed: float = 0.0
        self._count: int = 0

    def start(self):
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        elapsed = time.perf_counter() - self._start
        self._elapsed += elapsed
        self._count += 1
        self._start = None
        return elapsed

    def reset(self):
        self._elapsed = 0.0
        self._count = 0
        self._start = None

    @property
    def result(self) -> TimingResult:
        return TimingResult(self.name, self._elapsed, self._count)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class PipelineProfiler:
    def __init__(self):
        self.timers: dict[str, PrecisionTimer] = {}

    def get_timer(self, name: str) -> PrecisionTimer:
        if name not in self.timers:
            self.timers[name] = PrecisionTimer(name)
        return self.timers[name]

    def time_function(self, name: str, func, *args, **kwargs):
        timer = self.get_timer(name)
        timer.start()
        result = func(*args, **kwargs)
        timer.stop()
        return result

    def summary(self) -> list[TimingResult]:
        return [t.result for t in self.timers.values()]

    def print_summary(self):
        results = self.summary()
        if not results:
            print("No timing data.")
            return

        print("\n" + "=" * 60)
        print(f"{'Stage':<25} {'Avg (ms)':>10} {'FPS':>10} {'Runs':>8}")
        print("-" * 60)

        total_ms = 0
        for r in results:
            print(f"{r.name:<25} {r.avg_ms:>10.2f} {r.fps:>10.1f} {r.iterations:>8}")
            total_ms += r.avg_ms

        print("-" * 60)
        total_fps = 1000.0 / total_ms if total_ms > 0 else float("inf")
        print(f"{'TOTAL PIPELINE':<25} {total_ms:>10.2f} {total_fps:>10.1f}")
        print("=" * 60)
