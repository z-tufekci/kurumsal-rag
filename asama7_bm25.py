r"""Aşama 7 — Sparse Arama (BM25, SQLite FTS5).

Bu dosya bir İSKELETTİR: TODO'ları doldurarak tamamlayın. Adım numaraları
gelistirme_odevleri.md'deki Ödev 2'nin "Adımlar" listesiyle birebir
eşleşir — takıldığınızda oraya bakın, orada NEDEN sorusunun cevabı var.

Kavram: Dense (embedding) arama ANLAMA bakar — kelimeler farklı olsa bile
yakın anlamlıları bulur. BM25 ise tam tersi bir mantıkla çalışan klasik
bir arama yöntemidir: KELİME EŞLEŞMESİNE bakar. FTS5, SQLite'ın içine
gömülü, ek kütüphane gerektirmeyen tam metin arama eklentisidir — BM25'i
bunun üzerinden kuracaksınız.

Hedef sorularınızı bulmak için (yok tipi sorular farklı bir problemdir,
bu filtre onları eler):
    python degerlendirme.py --ayrinti | grep -E -A2 "\(tekil\)|\(sayisal\)|\(es_anlamli\)|\(coklu_dokuman\)"

ÖNEMLİ BEKLENTİ: Bu dosya kendi başına çalışır ama degerlendirme.py'nin
GENEL isabet oranını DEĞİŞTİRMEZ — bu normaldir, hata değildir. BM25
henüz asama5'teki ara() işlevine bağlı değil, yalnızca burada izole
test ediliyor. Toplam sayıya etkisini Ödev 3'te (Hybrid + RRF)
göreceksiniz.

Çalıştır: python asama7_bm25.py   (önce asama4 ile indeks kurulmuş olmalı)
"""

import re
import sqlite3

from asama2_parcalama import kaynak_etiketi
from asama4_vektor_deposu import INDEKS_KLASORU, indeks_yukle
from asama5_getirme import sorguyu_genislet

BM25_DB = INDEKS_KLASORU / "bm25.db"


def bm25_indeksi_kur(parcalar: list[dict]) -> None:
    """FTS5 sanal tablosunu kurup tüm parçaları ekler."""
    # Adım 2: sqlite3.connect(BM25_DB) ile AYRI bir veritabanı açın
    #   (vektör indeksinden bağımsız).

    # Adım 3: FTS5 sanal tablosu kurun:
    #   CREATE VIRTUAL TABLE parcalar_fts USING fts5(
    #       metin, tokenize='unicode61 remove_diacritics 2')

    # Adım 4: parcalar listesini SIRAYLA ekleyin —
    #   INSERT INTO parcalar_fts (rowid, metin) VALUES (?, ?)
    #   rowid'yi i+1 yapın (parcalar[rowid-1] ile geri dönebilmek için
    #   kritik: parça listesindeki konumla birebir eşleşmeli).

    # Bağlantıyı commit edip kapatmayı unutmayın.
    raise NotImplementedError


def bm25_ara(soru: str, parcalar: list[dict], k: int = 4) -> list[dict]:
    """Soruyu kelimelere ayırıp FTS5 üzerinden BM25 ile arar."""
    # Adım 5: soruyu re.findall(r"\w+", soru) ile kelimelere ayırın; FTS
    #   sorgusunu "kelime1* OR kelime2* OR ..." biçiminde kurun (* önek
    #   operatörü — Türkçe eklerde tam eşleşme arama başarısız olur).

    # Adım 6: SELECT rowid, bm25(parcalar_fts) FROM parcalar_fts
    #   WHERE parcalar_fts MATCH ? ORDER BY bm25(parcalar_fts) LIMIT ?
    #   çalıştırın. DİKKAT: bm25() skoru KÜÇÜK olan daha iyidir (negatif
    #   değerler) — ORDER BY artan sırada bırakın, dense ile ters mantık.

    # Adım 7: rowid -> parcalar[rowid-1] ile asıl parçaya dönün; sonuç
    #   listesine {**parca, "skor": bm25_skoru} olarak ekleyin.
    raise NotImplementedError


if __name__ == "__main__":
    _, parcalar = indeks_yukle()
    bm25_indeksi_kur(parcalar)

    # Adım 1: "python degerlendirme.py --ayrinti" çalıştırıp dense
    # aramanın kaçırdığı soruları KENDİNİZ bulun, buraya ekleyin.
    hedef_sorular = [
    ]

    for soru in hedef_sorular:
        genis = sorguyu_genislet(soru)
        print(f"\nSoru: {soru}")
        if genis != soru:
            print(f"  (genişletilmiş: {genis})")
        for s in bm25_ara(genis, parcalar):
            print(f"  {s['skor']:.2f}  {kaynak_etiketi(s)}")
