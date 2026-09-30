#!/usr/bin/env python
"""Extract pixel-tier and embedding-tier features for a directory of PNGs.

Both rungs are cached to one .npz per input directory so that every later
comparison -- query vs reference, clean vs perturbed -- is a cheap dot product
rather than a re-extraction.  The pixel features are produced by the original
author's own unmodified code, so the pixel rung in this study is literally the
published method, not a reimplementation of it.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "pixel"))


def lbp_fixed(img, P=8, R=1):
    """Uniform-LBP histogram at its canonical fixed length.

    The original `calculate_lbp_features` sizes its histogram as `lbp.max()+1`,
    which varies per image, so vectors are not comparable across pages and the
    published similarity raises on some pairs.  Uniform LBP with P neighbours
    has exactly P+2 patterns, so that is the length used here.  The author's
    file stays byte-identical; this wrapper is the repaired variant.
    """
    import cv2
    from skimage.feature import local_binary_pattern

    g = cv2.resize(np.array(img.convert("L")), (128, 128))
    lbp = local_binary_pattern(g, P, R, "uniform")
    h, _ = np.histogram(lbp.ravel(), bins=P + 2, range=(0, P + 2), density=True)
    return h.astype(np.float32)


def pixel_features(paths, roi_method="content_aware"):
    import image_processing as ip
    import config

    out = {"hash": [], "hw": [], "color": [], "lbp": [], "hog": [],
           "roi_hash": [], "roi_hw": [], "roi_color": [], "roi_lbp": [], "roi_hog": [],
           "gray256": [], "triv8": []}
    for i, p in enumerate(paths):
        img = Image.open(p)
        h, w_ = ip.calculate_multiresolution_hash_from_pil(img, **config.HASH_CONFIG)
        out["hash"].append(h.astype(np.uint8)); out["hw"].append(w_.astype(np.float32))
        out["color"].append(ip.calculate_color_histogram(img).ravel().astype(np.float32))
        out["lbp"].append(lbp_fixed(img))
        out["hog"].append(ip.calculate_hog_features(img).astype(np.float32))

        roi = ip.get_roi(img, method=roi_method)
        rh, rw = ip.calculate_multiresolution_hash_from_pil(roi, **config.HASH_CONFIG)
        out["roi_hash"].append(rh.astype(np.uint8)); out["roi_hw"].append(rw.astype(np.float32))
        out["roi_color"].append(ip.calculate_color_histogram(roi).ravel().astype(np.float32))
        out["roi_lbp"].append(lbp_fixed(roi))
        out["roi_hog"].append(ip.calculate_hog_features(roi).astype(np.float32))

        g = np.asarray(img.convert("L").resize((256, 256), Image.Resampling.LANCZOS),
                       dtype=np.float32)
        out["gray256"].append(g)
        # TRIV-8: the 8x8 mean-RGB thumbnail the audit found beats the whole fusion
        out["triv8"].append(np.asarray(
            img.convert("RGB").resize((8, 8), Image.Resampling.BOX),
            dtype=np.float32).ravel())

        if (i + 1) % 200 == 0:
            print(f"  pixel {i+1}/{len(paths)}", file=sys.stderr, flush=True)
    packed = {}
    for k, v in out.items():
        lens = {np.asarray(x).shape for x in v}
        if len(lens) != 1:
            raise ValueError(f"{k}: ragged feature, shapes={sorted(lens)[:4]}")
        packed[k] = np.stack(v)
    return packed


def clip_features(paths, batch=32):
    import open_clip
    import torch

    torch.set_num_threads(10)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai", device="cpu")
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            ims = torch.stack([preprocess(Image.open(p).convert("RGB"))
                               for p in paths[i:i + batch]])
            f = model.encode_image(ims)
            embs.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32))
            if (i // batch) % 10 == 0:
                print(f"  clip {i}/{len(paths)}", file=sys.stderr, flush=True)
    return np.concatenate(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roi", default="content_aware")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-pixel", action="store_true")
    ap.add_argument("--skip-clip", action="store_true")
    a = ap.parse_args()

    paths = sorted(Path(a.dir).glob("*.png"))
    if a.limit:
        paths = paths[: a.limit]
    ids = np.array([p.stem for p in paths])
    print(f"{len(paths)} görsel  <- {a.dir}", file=sys.stderr)

    store = {"ids": ids}
    if not a.skip_pixel:
        t = time.time()
        store.update({f"px_{k}": v for k, v in pixel_features(paths, a.roi).items()})
        print(f"piksel {time.time()-t:.0f}s ({(time.time()-t)/max(1,len(paths))*1000:.0f} ms/img)",
              file=sys.stderr)
    if not a.skip_clip:
        t = time.time()
        store["clip"] = clip_features(paths)
        print(f"clip   {time.time()-t:.0f}s ({(time.time()-t)/max(1,len(paths))*1000:.0f} ms/img)",
              file=sys.stderr)

    np.savez_compressed(a.out, **store)
    print(f"-> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
