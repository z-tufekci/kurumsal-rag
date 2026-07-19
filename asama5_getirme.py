"""Aşama 5 — Getirme (Retrieval).

RAG'ın kalbi: on binlerce parça içinden soruya en uygun 4-5 parçayı bulmak.
LLM belgeleri bilmez; yanlış parça getirirsek LLM yanlış bağlamla çalışır.
RAG kalitesinin tavanını retrieval belirler.

Bu naive sürüm yalnızca DENSE (anlamsal) arama yapar:

    Soru → embedding → tüm parça vektörleriyle kosinüs benzerliği
         → en yüksek TOP_K → eşik (threshold) altındakileri ele

Sparse arama (BM25), hybrid + RRF ve MMR geliştirme ödevidir.

ÖNEMLİ ALIŞKANLIK: Retrieval'ı LLM'e bağlamadan ÖNCE tek başına test edin
(bu dosyayı çalıştırın). Yanlış parça geliyorsa sorun LLM'de değil,
bu katmandadır — LLM'siz test etmek hatayı doğru yerde aratır.

Çalıştır:  python asama5_getirme.py   (önce asama4 ile indeks kurulmuş olmalı)
"""

import numpy as np

from asama2_parcalama import kaynak_etiketi
from asama3_gomme import kosinus_benzerligi, metinleri_gom
from asama4_vektor_deposu import indeks_yukle

TOP_K = 4    # LLM'e gönderilecek en fazla parça sayısı
ESIK = 0.45  # benzerlik eşiği (distance threshold): altı "alakasız" sayılır.
             # BGE-M3'te alakasız metinler bile ~0.40 benzerlik alabildiği için
             # 0.45 seçildi; değer degerlendirme.py deneyleriyle bulundu (0.35'e
             # göre isabet aynı, yanlış "alakalı" sayma daha az). Kurcalayın!


def ara(soru: str, vektorler: np.ndarray, parcalar: list[dict],
        k: int = TOP_K, esik: float = ESIK) -> list[dict]:
    """Soruya en benzer k parçayı bulur; eşik altında kalanları eler.

    Eşik olmasa en alakasız soruda bile "en yakın" k parça dönerdi —
    yakın olmak alakalı olmak demek değildir.
    """
    soru_vektoru = metinleri_gom([soru])
    skorlar = kosinus_benzerligi(soru_vektoru, vektorler)[0]
    sirali = np.argsort(skorlar)[::-1][:k]
    sonuclar = []
    for i in sirali:
        if skorlar[i] < esik:
            break  # skorlar azalan sıralı: ilk eşik altı sonuçta durabiliriz
        sonuclar.append({**parcalar[i], "skor": float(skorlar[i])})
    return sonuclar


if __name__ == "__main__":
    vektorler, parcalar = indeks_yukle()
    print(f"İndeks: {len(parcalar)} parça. Retrieval testi — çıkmak için boş satır.")
    print("Getirilen her parçanın TAM metni gösterilir — Aşama 6'da LLM'e gidecek")
    print("bağlam birebir budur; parça boyunun etkisini buradan gözlemleyin.\n")
    while True:
        soru = input("Soru: ").strip()
        if not soru:
            break
        sonuclar = ara(soru, vektorler, parcalar)
        if not sonuclar:
            print(f"  Eşik ({ESIK}) üstünde sonuç yok — soru dokümanlarla alakasız olabilir.")
        for i, s in enumerate(sonuclar, start=1):
            print(f"\n─── {i}. parça · skor {s['skor']:.3f} · {kaynak_etiketi(s)} "
                  f"· {len(s['metin'])} karakter ───")
            print(s["metin"])
        print()
