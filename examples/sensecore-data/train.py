"""65M byte-level LM smoke: mounted data, atomic checkpoint and exact resume.

This checks delivery/training/recovery, not pretraining quality or throughput.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width=672, heads=12):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x):
        b, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.projection(attention.transpose(1, 2).reshape(b, length, width))
        return x + self.mlp(self.norm2(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 672)
        self.blocks = nn.Sequential(*(Block() for _ in range(12)))
        self.norm = nn.LayerNorm(672)

    def forward(self, x):
        return F.linear(self.norm(self.blocks(self.embedding(x))), self.embedding.weight)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as body:
        while chunk := body.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/inputs/fineweb")
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--checkpoint-step", type=int, default=4)
    parser.add_argument("--pause-after-checkpoint", type=int, default=0)
    args = parser.parse_args()
    output = Path(os.environ["OUTPUT_DIR"])
    assert Path("/outputs").resolve() == output.resolve()
    assert torch.cuda.is_available(), "CUDA is required; CPU fallback is forbidden"
    assert not any(k.startswith("ML_EXPD_") for k in os.environ), "transfer credentials leaked to user process"
    import colorama
    assert colorama.__version__ == "0.4.6"
    torch.set_num_threads(4)
    torch.manual_seed(20261005)
    torch.cuda.manual_seed_all(20261005)
    model = Model().cuda().to(torch.bfloat16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
    parameters = sum(p.numel() for p in model.parameters())
    assert 65_000_000 <= parameters <= 66_000_000
    tokens = torch.tensor(list((Path(args.data) / "fineweb-sample.txt").read_bytes()), dtype=torch.long, device="cuda")
    start_step, restored_sha = 0, None
    if args.resume:
        path = Path(args.resume)
        restored_sha = sha(path)
        state = torch.load(path, map_location="cuda", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        torch.cuda.set_rng_state(state["rng_cuda"].cpu())
        start_step = state["step"]
        assert state["parameters"] == parameters
    runtime = {"gpu": torch.cuda.get_device_name(0), "parameters": parameters,
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "numpy": importlib.metadata.version("numpy"), "colorama": colorama.__version__,
               "source_id": os.environ["SOURCE_ID"], "run_id": os.environ["RUN_ID"],
               "attempt_id": os.environ["ATTEMPT_ID"], "start_step": start_step, "restored_sha256": restored_sha,
               "data_text_sha256": sha(Path(args.data) / "fineweb-sample.txt"),
               "client_dockerfile_proof": json.loads(Path("/usr/local/share/user-dockerfile-proof.json").read_text())}
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print("RUNTIME " + json.dumps(runtime), flush=True)
    begin = time.monotonic()
    with (output / "metrics.jsonl").open("w") as metrics:
        for step in range(start_step + 1, args.steps + 1):
            indexes = torch.arange(4 * 129, device="cuda").reshape(4, 129)
            indexes = (indexes + step * 513) % len(tokens)
            batch = tokens[indexes]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch[:, :-1]).float().reshape(-1, 256), batch[:, 1:].reshape(-1))
            assert math.isfinite(float(loss))
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            row = {"step": step, "loss": float(loss), "elapsed_seconds": time.monotonic() - begin}
            metrics.write(json.dumps(row) + "\n"); metrics.flush()
            print("METRIC " + json.dumps(row), flush=True)
            if step == args.checkpoint_step or step == args.steps:
                checkpoint = output / f"checkpoints/step-{step:04d}/state.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".tmp")
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                            "parameters": parameters, "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, temporary)
                temporary.replace(checkpoint)
                ready = {"step": step, "files": [{"path": checkpoint.relative_to(output).as_posix(),
                          "sha256": sha(checkpoint), "bytes": checkpoint.stat().st_size}]}
                marker = output / "checkpoint.tmp"
                marker.write_text(json.dumps(ready) + "\n")
                marker.replace(output / "checkpoint.ready.json")
                print("CHECKPOINT_READY " + json.dumps(ready), flush=True)
                if step == args.checkpoint_step and args.pause_after_checkpoint:
                    time.sleep(args.pause_after_checkpoint)
    result = {**runtime, "final_step": args.steps, "elapsed_seconds": time.monotonic() - begin,
              "final_loss": float(loss), "checkpoint_sha256": ready["files"][0]["sha256"], "passed": True}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
