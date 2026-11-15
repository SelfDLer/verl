"""Capture one actor microbatch on its executing NPU rank."""
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch_npu import profiler


@contextmanager
def capture(directory, samples):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    torch.npu.synchronize()
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        experimental_config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level1
        ),
        on_trace_ready=profiler.tensorboard_trace_handler(str(path), analyse_flag=False),
    ):
        yield
        torch.npu.synchronize()
    (path / "capture.json").write_text(json.dumps({
        "global_step": 3, "rank": torch.distributed.get_rank(), "microbatch": 0,
        "samples": int(samples), "scope": "actor forward and backward",
    }))
