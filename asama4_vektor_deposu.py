"""Aşama 4 — Vektör Deposu (Vector Store).

Naive depo: numpy matrisi + JSON metadata, diske kaydedilir.

Neden diske kaydediyoruz (persistence)? Stajyer prototipindeki
`VectorStoreIndex.from_documents()` indeksi RAM'de kuruyordu: program
kapanınca indeks yok oluyor, her açılışta tüm belgeler YENİDEN gömülüyordu.
Binlerce belgede bu saatler sürer. Bir kez kur, diske yaz, sonra yükle.

Gerçek projelerde bu katman Chroma/Qdrant gibi bir vektör veritabanıdır
(metadata filtreleme, artımlı güncelleme, ölçek). Chroma'ya geçiş
geliştirme ödevidir — bu dosyadaki iki fonksiyonun imzası korunursa
diğer aşamalar hiç değişmeden çalışmaya devam eder. Kurumsal projede
vectorstore.py'nin "tek arayüz" olması da aynı fikirdir.

Çalıştır:  python asama4_vektor_deposu.py
"""

import json
import time
from pathlib import Path

import numpy as np

from asama1_belge_okuma import belgeleri_oku
from asama2_parcalama import belgeleri_parcala
from asama3_gomme import metinleri_gom

INDEKS_KLASORU = Path(__file__).parent / "indeks"


def indeks_kur() -> None:
    """Belgeleri okur, parçalar, gömer ve indeksi diske kaydeder.

    Naive davranış: her çağrıda HER ŞEYİ baştan gömer. Yalnızca değişen
    dosyaları gömmek (artımlı indeksleme, örn. SHA-256 ile) geliştirme ödevidir.
    """
    parcalar = belgeleri_parcala(belgeleri_oku())
    print(f"{len(parcalar)} parça gömülüyor (embedding)...")
    baslangic = time.time()
    vektorler = metinleri_gom([p["metin"] for p in parcalar])
    print(f"Bitti: {time.time() - baslangic:.1f} sn")

    INDEKS_KLASORU.mkdir(exist_ok=True)
    np.save(INDEKS_KLASORU / "vektorler.npy", vektorler)
    (INDEKS_KLASORU / "parcalar.json").write_text(
        json.dumps(parcalar, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"İndeks kaydedildi: {INDEKS_KLASORU}/")


def indeks_yukle() -> tuple[np.ndarray, list[dict]]:
    """Kayıtlı indeksi diskten yükler — gömme maliyeti ödenmez."""
    vektorler = np.load(INDEKS_KLASORU / "vektorler.npy")
    parcalar = json.loads(
        (INDEKS_KLASORU / "parcalar.json").read_text(encoding="utf-8")
    )
    return vektorler, parcalar


if __name__ == "__main__":
    indeks_kur()
    vektorler, parcalar = indeks_yukle()
    print(f"Doğrulama: {vektorler.shape[0]} vektör × {vektorler.shape[1]} boyut, "
          f"{len(parcalar)} parça metadata'sı")
