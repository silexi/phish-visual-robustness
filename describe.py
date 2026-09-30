#!/usr/bin/env python
"""Structured page description with a Qwen3-VL model under MLX.

Gate test: does a VLM recover the semantics of a page whose *text* has been
obfuscated to defeat text/NLP detectors?  Writes one JSON record per image.
"""
import argparse, json, sys, time
from pathlib import Path

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

PROMPT = """You are analysing a screenshot of a web page for a security study.

Return ONLY a JSON object, no prose, no markdown fence, with exactly these keys:

{
  "brand": "the company/service this page belongs to, or null if unclear",
  "page_type": "one of: login, mfa_challenge, password_entry, account_recovery, landing, error, other",
  "asks_for_credentials": true or false,
  "visible_text": ["every piece of text you can read, VERBATIM, preserving the exact spacing and spelling you actually see - do not correct, normalise or de-space anything"],
  "input_fields": ["label or placeholder of each input box"],
  "buttons": ["text on each button"],
  "logo_present": true or false,
  "logo_description": "what the logo looks like, or null",
  "dominant_colors": ["up to 4 colour names"],
  "layout": "one short sentence on the visual layout"
}

The "visible_text" field is the important one: transcribe characters exactly as
rendered, including any unusual spacing inside words. Do not repair the text."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-8bit")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default="descriptions.jsonl")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--resize", type=int, default=1024,
                    help="longest side in px before the model sees it; 0 = native")
    args = ap.parse_args()

    t0 = time.time()
    print(f"loading {args.model} ...", file=sys.stderr, flush=True)
    model, processor = load(args.model)
    config = model.config
    print(f"loaded in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)

    out = Path(args.out)
    n_ok = 0
    with out.open("w", encoding="utf-8") as fh:
        for path in args.images:
            p = Path(path)
            img = str(p)

            if args.resize:
                from PIL import Image
                im = Image.open(p).convert("RGB")
                im.thumbnail((args.resize, args.resize), Image.LANCZOS)
                tmp = p.parent / f".resized_{p.stem}.png"
                im.save(tmp)
                img = str(tmp)

            prompt = apply_chat_template(processor, config, PROMPT, num_images=1)

            t = time.time()
            res = generate(model, processor, prompt, image=[img],
                           max_tokens=args.max_tokens, temperature=args.temp,
                           verbose=False)
            dt = time.time() - t
            text = res.text if hasattr(res, "text") else str(res)

            raw = text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                raw = raw[4:] if raw.lower().startswith("json") else raw
            try:
                parsed, err = json.loads(raw.strip()), None
                n_ok += 1
            except Exception as e:
                parsed, err = None, f"{type(e).__name__}: {e}"

            rec = {
                "image": p.name,
                "seconds": round(dt, 2),
                "prompt_tokens": getattr(res, "prompt_tokens", None),
                "generation_tokens": getattr(res, "generation_tokens", None),
                "tokens_per_second": round(getattr(res, "generation_tps", 0) or 0, 1),
                "peak_memory_gb": round(getattr(res, "peak_memory", 0) or 0, 2),
                "parsed": parsed,
                "parse_error": err,
                "raw": None if parsed else text,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

            brand = (parsed or {}).get("brand")
            print(f"  {p.name:<42} {dt:6.1f}s  {rec['tokens_per_second']:>5.1f} tok/s  "
                  f"brand={brand}", file=sys.stderr, flush=True)

    print(f"\n{n_ok}/{len(args.images)} parsed as JSON -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
