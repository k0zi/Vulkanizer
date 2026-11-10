# One Z-Image Turbo model forward at 256x256 through ComfyUI, on the Vulkanizer GPU (fp16, as ComfyUI
# runs it) and on the CPU in fp32 and in fp16, from the same conditioning and noise. The GPU result
# is correct if it is as far from fp32 as fp16 itself is, and close to the CPU's fp16 (the rest is
# summation order). Needs a ComfyUI checkout with its venv-vulkanizer environment and the Z-Image
# Turbo models (README, "ComfyUI"); takes a few minutes and ~25 GB RAM.
#
#   python3 tests/torch/zimage_forward.py [--comfy DIR] [--env DIR] [--out DIR]
#
# The steps run in their own processes (`zimage_forward.py STEP OUT`, from the ComfyUI directory):
#   cond   text encoder on the CPU (fp32) -> OUT/cond.pt
#   gpu16  diffusion model on the CUDA device (through run_torch.sh), fp16
#   cpu32  diffusion model on the CPU, fp32
#   cpu16  diffusion model on the CPU, fp16 weights and math (ComfyUI would cast to fp32 there)
# Each model step writes OUT/<step>.pt: the denoised latent of one euler step from sigma 1, i.e.
# exactly one model evaluation.
import os, subprocess, sys, time
from pathlib import Path

STEPS = ("cond", "gpu16", "cpu32", "cpu16")
PROMPT = ("Latina female with thick wavy hair, harbor boats and pastel houses behind. "
          "Breezy seaside light, warm tones, cinematic close-up. ")


def run_step(step, out):
    flags = ["--disable-dynamic-vram"] + (["--disable-cuda-malloc"] if step == "gpu16" else ["--cpu"])
    sys.argv = ["main.py"] + flags
    sys.path.insert(0, os.getcwd())
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import comfy.model_management as mm
    import comfy.sd
    import folder_paths
    import nodes
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

    if step == "cond":
        clip = nodes.CLIPLoader().load_clip("qwen_3_4b.safetensors", "lumina2", "default")[0]
        pos = nodes.CLIPTextEncode().encode(clip, PROMPT)[0]
        neg = [[torch.zeros_like(c[0]), dict(c[1])] for c in pos]  # ConditioningZeroOut
        torch.save({"pos": pos, "neg": neg}, f"{out}/cond.pt")
        return
    if step == "cpu16":
        mm.unet_manual_cast = lambda *a, **k: None  # compute in the weight dtype on the CPU too
    dtype = torch.float32 if step == "cpu32" else torch.float16
    path = folder_paths.get_full_path_or_raise("diffusion_models", "z_image_turbo_bf16.safetensors")
    model = comfy.sd.load_diffusion_model(path, model_options={"dtype": dtype})
    model = ModelSamplingAuraFlow().patch_aura(model, 3)[0]
    print(f"{step}: {model.load_device}, weights {model.model.get_dtype()}, manual cast {model.model.manual_cast_dtype}",
          flush=True)
    cond = torch.load(f"{out}/cond.pt", weights_only=False)
    latent = {"samples": torch.zeros(1, 16, 32, 32), "downscale_ratio_spacial": 8}  # EmptySD3LatentImage 256x256
    res = nodes.common_ksampler(model, 42, 1, 1.0, "euler", "simple", cond["pos"], cond["neg"], latent, denoise=1.0)[0]
    torch.save(res["samples"].float().cpu(), f"{out}/{step}.pt")


def compare(out):
    import torch
    r = {s: torch.load(f"{out}/{s}.pt").double() for s in STEPS[1:]}
    rel = lambda a, b: ((r[a] - r[b]).norm() / r[b].norm()).item()
    cos = lambda a, b: torch.nn.functional.cosine_similarity(r[a].flatten(), r[b].flatten(), dim=0).item()
    for a, b in (("gpu16", "cpu32"), ("cpu16", "cpu32"), ("gpu16", "cpu16")):
        print(f"{a} vs {b}: rel L2 {rel(a, b):.2e}, cosine {cos(a, b):.6f}")
    fp16 = rel("cpu16", "cpu32")
    ok = rel("gpu16", "cpu32") <= 1.5 * fp16 and rel("gpu16", "cpu16") < fp16
    print("OK: the GPU differs from fp32 as much as fp16 does" if ok else "FAIL: the GPU differs more than fp16 explains")
    return ok


def main():
    args = sys.argv[1:]
    opt = lambda name, default: args[args.index(name) + 1] if name in args else default
    root = Path(__file__).resolve().parents[2]
    comfy = Path(opt("--comfy", Path.home() / "Develop/GitHub/ComfyUI")).resolve()
    env = Path(opt("--env", comfy / "venv-vulkanizer")).resolve()
    out = Path(opt("--out", root / ".work/zimage_forward")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    python = str(env / "bin/python")
    for step in STEPS:
        cmd = [python, __file__, step, str(out)]
        if step == "gpu16":
            cmd = [str(root / "run_torch.sh"), "--env", str(env), __file__, step, str(out)]
        t0 = time.time()
        if subprocess.run(cmd, cwd=comfy).returncode != 0 or not (out / f"{step}.pt").exists():
            sys.exit(f"step {step} failed")
        print(f"{step}: {time.time() - t0:.0f} s", flush=True)
    sys.exit(0 if compare(out) else 1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in STEPS:
        step, out = sys.argv[1], sys.argv[2]
        run_step(step, out)
        os._exit(0)  # skip interpreter teardown (ComfyUI's atexit handlers fail there)
    main()
