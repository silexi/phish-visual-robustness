#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JNCA gonderimi icin sekilleri uretir.

Sekiller INGILIZCE uretilir; makale gonderim oncesi ingilizceye cevrilecegi
icin etiketlerin bastan hedef dilde olmasi yeniden uretimi gereksiz kilar.
Hoca icin Turkce ara surum gerekirse:  python sekil_uret.py tr

Elsevier kisitlari:
  genislik   tek sutun 90 mm | 1,5 sutun 140 mm | tam genislik 190 mm
  cozunurluk fotografik 300 dpi, cizgi/birlesik 500-1000 dpi (vektorde konusuz)
  yazi tipi  Arial/Helvetica/Courier/Times, gomulu, basili boyut >= 7 pt
Grafikler vektor PDF, ekran goruntusu bilesikleri 400 dpi PNG olarak yazilir.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

DIL = "tr" if len(sys.argv) > 1 and sys.argv[1] == "tr" else "en"

S = Path(__file__).parent / "input"
CIK = Path(__file__).parent / "out"
if DIL == "tr":
    CIK = CIK / "tr"
CIK.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
TEK, BIRBUCUK, TAM = 90 * MM, 140 * MM, 190 * MM

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,      # TrueType gomme; Type 3 yayincilarca reddedilir
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})

M = {
    "en": {
        "clean": "(a) clean render",
        "A3": "(b) A3  filter:invert(1)",
        "A4": "(c) A4  decoy block",
        "S1": "(d) S1  images hidden",
        "S3": "(e) S3  letter spacing",
        "S7": "(f) S7  false brand footer",
        "x_ret": "cluster-level rank-1 retention",
        "x_cost": "perceptibility  $f_{\\mathrm{JND}}$  (median, log scale)",
        "y_ret": "retention",
        "leg_px": "fusion (pixel)",
        "leg_sem": "brand (semantic)",
        "s7a": "(a) clean render",
        "s7b": "(b) S7: a single footer line added to the page",
        "ans": "model answer:  brand = \u201c{}\u201d",
        "rungs": {"P_color": "colour", "P5_triv8": "8\u00d78",
                  "P_ssim_global": "SSIM*", "P3_hash": "hash", "P_lbp": "LBP",
                  "P0_fusion": "fusion", "E1_clip": "CLIP", "P4_hog": "HOG"},
        "ondalik": False,
    },
    "tr": {
        "clean": "(a) temiz render",
        "A3": "(b) A3  filter:invert(1)",
        "A4": "(c) A4  yem bloğu",
        "S1": "(d) S1  görseller gizli",
        "S3": "(e) S3  harf aralığı",
        "S7": "(f) S7  sahte marka altbilgisi",
        "x_ret": "küme düzeyinde ilk sıra korunma oranı",
        "x_cost": "algısal fark edilebilirlik  $f_{\\mathrm{JND}}$  (medyan, log ölçek)",
        "y_ret": "korunma oranı",
        "leg_px": "füzyon (piksel)",
        "leg_sem": "marka (anlamsal)",
        "s7a": "(a) temiz render",
        "s7b": "(b) S7: sayfaya tek satır altbilgi eklendi",
        "ans": "model yanıtı:  marka = \u201c{}\u201d",
        "rungs": {"P_color": "renk", "P5_triv8": "8\u00d78",
                  "P_ssim_global": "SSIM*", "P3_hash": "hash", "P_lbp": "LBP",
                  "P0_fusion": "füzyon", "E1_clip": "CLIP", "P4_hog": "HOG"},
        "ondalik": True,
    },
}[DIL]


def ond(x, _=None):
    """Turkce surumde ondalik ayirici virgul olmali."""
    return f"{x:g}".replace(".", ",")


def bicimle(ax, eksen):
    if M["ondalik"]:
        getattr(ax, eksen + "axis").set_major_formatter(
            matplotlib.ticker.FuncFormatter(ond))


def say(v, basamak=3):
    s = f"{v:.{basamak}f}"
    return s.replace(".", ",") if M["ondalik"] else s


ret = json.loads((S / "results_retention_v2.json").read_text(encoding="utf-8"))
perc = json.loads((S / "results_perceptibility.json").read_text(encoding="utf-8"))
vlm = json.loads((S / "results_vlm_v2.json").read_text(encoding="utf-8"))


# =====================================================================
# Sekil 1 -- bozulma kosullarinin gorsel karsiligi (2x3 panel, tam genislik)
# =====================================================================
def sekil1():
    P = S / "panel"
    paneller = [("clean.png", M["clean"]), ("A3.png", M["A3"]),
                ("A4.png", M["A4"]), ("S1.png", M["S1"]),
                ("S3.png", M["S3"]), ("S7.png", M["S7"])]
    ims = [Image.open(P / a) for a, _ in paneller]
    w, h = ims[0].size
    sut, sat = 3, 2
    panel_w = TAM / sut
    panel_h = panel_w * h / w

    fig, axes = plt.subplots(sat, sut, figsize=(TAM, sat * panel_h + sat * 0.16))
    for ax, im, (_, etiket) in zip(axes.ravel(), ims, paneller):
        ax.imshow(np.asarray(im), interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
            s.set_color("#888888")
        ax.set_title(etiket, fontsize=7, pad=2.5)
    fig.subplots_adjust(wspace=0.03, hspace=0.20, left=0, right=1, top=0.94,
                        bottom=0)
    fig.savefig(CIK / "fig1_conditions.png", dpi=400)
    fig.savefig(CIK / "fig1_conditions.pdf")
    plt.close(fig)
    print(f"  fig1: panel {panel_w / MM:.0f} mm, {w} px -> "
          f"{w / (panel_w):.0f} dpi")


# =====================================================================
# Sekil 2 -- ters cevirme altinda yapilandirma bazinda korunma + guven araligi
#            Tablo 3'te bulunmayan bilgi: her yapilandirma icin bootstrap CI
# =====================================================================
def sekil2():
    r = ret["conditions"]["A3"]["rungs"]
    veri = [(ad, r[k]["applied_only"]["cluster_mean"],
             r[k]["applied_only"]["boot95"][0], r[k]["applied_only"]["boot95"][1])
            for k, ad in M["rungs"].items()]
    veri.sort(key=lambda x: x[1])
    orta = np.array([v[1] for v in veri])
    alt = np.array([v[2] for v in veri])
    ust = np.array([v[3] for v in veri])
    y = np.arange(len(veri))

    fig, ax = plt.subplots(figsize=(TEK, 60 * MM))
    ax.hlines(y, alt, ust, color="#4a4a4a", lw=1.1, zorder=2)
    ax.plot(alt, y, "|", ms=4, color="#4a4a4a", zorder=2)
    ax.plot(ust, y, "|", ms=4, color="#4a4a4a", zorder=2)
    renk = ["#b2182b" if v < 0.3 else "#2166ac" if v > 0.6 else "#777777"
            for v in orta]
    ax.scatter(orta, y, s=26, c=renk, zorder=3, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(orta):
        ax.annotate(say(v), (v, i), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels([v[0] for v in veri])
    ax.set_xlim(-0.03, 1.05)
    bicimle(ax, "x")
    ax.set_xlabel(M["x_ret"])
    ax.grid(axis="x", lw=0.35, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(CIK / "fig2_inversion_ci.pdf")
    plt.close(fig)
    print(f"  fig2: yayilim {orta.min():.3f}-{orta.max():.3f}")


# =====================================================================
# Sekil 3 -- algisal fark edilebilirlik ile korunma iliskisi (iki kanal)
#            Tablo 3 ile Tablo 5'i tek eksende birlestirir.
# =====================================================================
def sekil3():
    kos = ["A1", "A3", "A4", "S1", "S3", "S7"]
    x = [perc[k]["f_JND"]["median"] for k in kos]
    piksel = [ret["conditions"][k]["rungs"]["P0_fusion"]["applied_only"]["cluster_mean"]
              for k in kos]
    marka = [vlm["conditions"][k]["brand"]["rate"] for k in kos]

    fig, ax = plt.subplots(figsize=(TEK, 62 * MM))
    ax.plot(x, piksel, "o", ms=5, color="#2166ac", label=M["leg_px"],
            zorder=3, mec="white", mew=0.5)
    ax.plot(x, marka, "^", ms=5, color="#b2182b", label=M["leg_sem"],
            zorder=3, mec="white", mew=0.5)
    # A4 (0,0460) ile S3 (0,0484) log eksende neredeyse ust uste; etiketleri
    # hem yana hem dikeyde kaydirarak hangi noktaya ait olduklari korunur
    KAY = {"A4": (-9, 5, "right"), "S3": (9, 14, "left")}
    for xi, p, m, k in zip(x, piksel, marka, kos):
        ax.plot([xi, xi], [min(p, m), max(p, m)], color="#cccccc", lw=0.7,
                zorder=1)
        dx, dy, ha = KAY.get(k, (0, 7, "center"))
        ax.annotate(k, (xi, max(p, m)), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel(M["x_cost"])
    ax.set_ylabel(M["y_ret"])
    bicimle(ax, "y")
    ax.set_ylim(0, 1.12)
    ax.grid(lw=0.35, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.4)
    fig.savefig(CIK / "fig3_perceptibility_retention.pdf")
    plt.close(fig)
    print(f"  fig3: en az fark edilen etkili kosul S1 "
          f"(f_JND={perc['S1']['f_JND']['median']}, "
          f"marka={vlm['conditions']['S1']['brand']['rate']})")


# =====================================================================
# Sekil 4 -- S7 mekanizmasi: temiz ve bozulmus render yan yana
#            Kirmizi cerceve sonradan eklenen bir ISARETTIR; goruntude
#            hicbir icerik gizlenmemis veya kaldirilmamistir.
# =====================================================================
def sekil4():
    P = S / "panel"
    a, b = Image.open(P / "clean.png"), Image.open(P / "S7.png")
    w, h = a.size
    panel_w = TAM / 2
    panel_h = panel_w * h / w

    fig = plt.figure(figsize=(TAM, panel_h + 0.40))
    for i, (im, ust, marka) in enumerate([
            (a, M["s7a"], "sharepoint"),
            (b, M["s7b"], "aventro corporation")]):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.imshow(np.asarray(im), interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
            s.set_color("#888888")
        ax.set_title(ust, fontsize=7, pad=2.5)
        ax.set_xlabel(M["ans"].format(marka), fontsize=7, labelpad=3)
        if i == 1:
            ax.add_patch(Rectangle((0, h * 0.945), w, h * 0.055, fill=False,
                                   edgecolor="#b2182b", lw=0.9))
    fig.subplots_adjust(wspace=0.04, left=0, right=1, top=0.90, bottom=0.10)
    fig.savefig(CIK / "fig4_s7_mechanism.png", dpi=400)
    fig.savefig(CIK / "fig4_s7_mechanism.pdf")
    plt.close(fig)
    print(f"  fig4: panel {panel_w / MM:.0f} mm -> {w / panel_w:.0f} dpi")


print(f"dil: {DIL}   cikti: {CIK}")
for f in (sekil1, sekil2, sekil3, sekil4):
    print(f.__name__)
    f()

print("\nuretilen dosyalar:")
for p in sorted(CIK.glob("*.*")):
    print(f"  {p.name:<28} {p.stat().st_size / 1024:8.0f} KB")
