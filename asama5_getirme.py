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

import re
from pathlib import Path

import numpy as np

from asama2_parcalama import kaynak_etiketi
from asama3_gomme import kosinus_benzerligi, metinleri_gom
from asama4_vektor_deposu import indeks_yukle

SOZLUK_DOSYASI = Path(__file__).parent / "dokumanlar" / "kisaltmalar_ve_terimler_sozlugu.txt"

TOP_K = 4    # LLM'e gönderilecek en fazla parça sayısı
ESIK = 0.45  # benzerlik eşiği (distance threshold): altı "alakasız" sayılır.
             # BGE-M3'te alakasız metinler bile ~0.40 benzerlik alabildiği için
             # 0.45 seçildi; değer degerlendirme.py deneyleriyle bulundu (0.35'e
             # göre isabet aynı, yanlış "alakalı" sayma daha az). Kurcalayın!


def _tr_buyuk(metin: str) -> str:
    """Türkçe'ye uygun büyük harfe çevirme: i→İ, ı→I (Python'un upper()'ı bunu bilmez)."""
    return metin.translate(str.maketrans("iı", "İI")).upper()


def kisaltma_sozlugu_yukle() -> dict[str, str]:
    """Sözlük dokümanındaki 'KISALTMA: Açılım.' satırlarından eşleme çıkarır.

    Eşlemeyi ayrı bir konfigürasyon dosyasında değil, korpustaki sözlük
    dokümanının kendisinden okuyoruz: tek doğruluk kaynağı. Sözlüğe eklenen
    her kısaltma, yeniden indekslemeye bile gerek kalmadan sorgu
    genişletmede anında geçerli olur.
    """
    eslesme = {}
    if not SOZLUK_DOSYASI.exists():
        return eslesme
    for satir in SOZLUK_DOSYASI.read_text(encoding="utf-8").splitlines():
        e = re.match(r"^([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9-]{1,9}): (.+)$", satir.strip())
        if e:
            eslesme[e.group(1)] = e.group(2).split(".")[0].strip()
    return eslesme


KISALTMALAR = kisaltma_sozlugu_yukle()


def sorguyu_genislet(soru: str) -> str:
    """Sorudaki kısaltmaların açılımlarını soruya ekler (query expansion).

    Problem: kullanıcı "KDİ'de ne denetlenir?" diye sorar ama dokümanlar
    "Karayolları Denetim İstasyonu" yazar — embedding kısaltmayla açılımı
    her zaman yakın gömemez. Çözüm: açılımı sorunun sonuna iliştiririz;
    embedding artık iki biçimi de görür. Bu, pipeline'daki "sorgu
    temizleme/zenginleştirme" (query cleaning) adımının en yalın halidir.
    """
    ekler = []
    for parca in re.findall(r"[A-Za-z0-9ÇĞİÖŞÜçğıöşü-]+", soru):
        anahtar = _tr_buyuk(parca)
        if anahtar in KISALTMALAR:
            ek = f"{anahtar}: {KISALTMALAR[anahtar]}"
            if ek not in ekler:
                ekler.append(ek)
    if not ekler:
        return soru
    return f"{soru} ({'; '.join(ekler)})"


def ara(soru: str, vektorler: np.ndarray, parcalar: list[dict],
        k: int = TOP_K, esik: float = ESIK) -> list[dict]:
    """Soruya en benzer k parçayı bulur; eşik altında kalanları eler.

    Arama öncesi soru, kısaltma açılımlarıyla genişletilir (bkz.
    sorguyu_genislet). Eşik olmasa en alakasız soruda bile "en yakın"
    k parça dönerdi — yakın olmak alakalı olmak demek değildir.
    """
    soru = sorguyu_genislet(soru)
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
        genis = sorguyu_genislet(soru)
        if genis != soru:
            print(f"↳ Sorgu genişletildi: {genis}")
        sonuclar = ara(soru, vektorler, parcalar)
        if not sonuclar:
            print(f"  Eşik ({ESIK}) üstünde sonuç yok — soru dokümanlarla alakasız olabilir.")
        for i, s in enumerate(sonuclar, start=1):
            print(f"\n─── {i}. parça · skor {s['skor']:.3f} · {kaynak_etiketi(s)} "
                  f"· {len(s['metin'])} karakter ───")
            print(s["metin"])
        print()
