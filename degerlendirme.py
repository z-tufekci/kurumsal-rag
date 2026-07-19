"""Değerlendirme — Retrieval İsabet Ölçümü (Evaluation / Hit Rate).

"Bence daha iyi oldu" bir gerekçe değildir; her geliştirme SAYIYLA savunulur.
Bu script, degerlendirme/sorular.csv'deki soruları retrieval katmanından
geçirir ve iki metrik üretir:

1. İsabet@k (hit rate): beklenen kaynak dosya, getirilen ilk k parçanın
   içinde mi? (Cevap kalitesini değil, DOĞRU PARÇANIN bulunmasını ölçer —
   retrieval'ı LLM'siz, izole ve hızlı test etmenin yolu.)
2. Ret testi: cevabı dokümanlarda OLMAYAN sorularda ("yok" tipi) eşik
   üstünde parça dönmemesi beklenir. Parça dönerse LLM'in 2. kuralı
   (bilmiyorum demek) son savunma hattı olarak devreye girer.

Kullanım (önce asama4 ile indeks kurulmuş olmalı):
    python degerlendirme.py                # varsayılan k ve eşik
    python degerlendirme.py --k 8          # top-k deneyi
    python degerlendirme.py --esik 0.5     # eşik deneyi
    python degerlendirme.py --ayrinti      # başarısız soruları listele

Geliştirme (hybrid, chunk ayarı, MMR...) yaptıkça bu scripti yeniden
çalıştırıp sayıları KARŞILAŞTIRIN — ödevlerin ölçüm aracı budur.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from asama4_vektor_deposu import indeks_yukle
from asama5_getirme import ESIK, TOP_K, ara

SORU_DOSYASI = Path(__file__).parent / "degerlendirme" / "sorular.csv"


def sorulari_oku() -> list[dict]:
    with open(SORU_DOSYASI, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def degerlendir(k: int, esik: float, ayrinti: bool) -> None:
    vektorler, parcalar = indeks_yukle()
    sorular = sorulari_oku()
    print(f"{len(sorular)} soru, k={k}, eşik={esik}\n")

    tip_toplam = defaultdict(int)
    tip_isabet = defaultdict(int)
    hatalar = []

    for soru in sorular:
        sonuclar = ara(soru["soru"], vektorler, parcalar, k=k, esik=esik)
        bulunan_dosyalar = {s["dosya"] for s in sonuclar}
        tip = soru["tip"]
        tip_toplam[tip] += 1

        if tip == "yok":
            basarili = not sonuclar  # hiç parça dönmemeli
        else:
            beklenen = set(soru["kaynak_dosyalar"].split(";"))
            basarili = bool(beklenen & bulunan_dosyalar)

        if basarili:
            tip_isabet[tip] += 1
        else:
            hatalar.append((soru, sorted(bulunan_dosyalar)))

    cevaplanabilir = [t for t in tip_toplam if t != "yok"]
    toplam = sum(tip_toplam[t] for t in cevaplanabilir)
    isabet = sum(tip_isabet[t] for t in cevaplanabilir)
    print(f"İsabet@{k} (cevaplanabilir {toplam} soru): {isabet}/{toplam} "
          f"= %{100 * isabet / toplam:.1f}")
    for tip in sorted(cevaplanabilir):
        print(f"  {tip:<15} {tip_isabet[tip]}/{tip_toplam[tip]}")
    if tip_toplam["yok"]:
        print(f"Ret testi ('yok' {tip_toplam['yok']} soru, boş dönüş beklenir): "
              f"{tip_isabet['yok']}/{tip_toplam['yok']}")
        print("  (Düşükse eşik fazla gevşek demektir; --esik ile deneyin. "
              "Kaçanları son savunma olarak LLM'in 'bilmiyorum' kuralı karşılar.)")

    if hatalar and ayrinti:
        print(f"\nBaşarısız {len(hatalar)} soru:")
        for soru, bulunan in hatalar:
            print(f"  [{soru['no']}] ({soru['tip']}) {soru['soru']}")
            print(f"      beklenen: {soru['kaynak_dosyalar'] or '(boş dönüş)'}")
            print(f"      gelen   : {';'.join(bulunan) or '(boş)'}")
    elif hatalar:
        print(f"\n{len(hatalar)} soru başarısız (--ayrinti ile listeleyin).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Retrieval isabet ölçümü")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--esik", type=float, default=ESIK)
    ap.add_argument("--ayrinti", action="store_true")
    args = ap.parse_args()
    degerlendir(args.k, args.esik, args.ayrinti)
