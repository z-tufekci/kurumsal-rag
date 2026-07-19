"""Aşama 3 — Gömme (Embedding).

Embedding, metni anlamını temsil eden bir sayı dizisine (vektöre) çevirir.
Anlamca yakın metinlerin vektörleri de uzayda yakın düşer — "araç plakası"
ile "taşıt kimliği" farklı kelimeler ama yakın vektörlerdir. Dense
(anlamsal) aramanın tamamı bu özelliğe dayanır.

İki ayrı model çalıştığımıza dikkat:
- Embedding modeli (BGE-M3)  → cevap YAZMAZ, sadece vektör üretir.
- LLM (Aşama 6'da)           → vektör üretmez, cevap yazar.

ALTIN KURAL: Sorgu ve belgeler AYNI embedding modeliyle gömülmek zorundadır.
Model değişirse tüm indeks baştan kurulur — kurumsal projede de bu böyledir.

Gereksinim: Ollama çalışıyor olmalı (`ollama pull bge-m3` yapılmış olmalı).

Çalıştır:  python asama3_gomme.py
"""

import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434"   # Ollama lokal API — veri makineden çıkmaz
EMBED_MODELI = "bge-m3"                 # kurumsal projeyle aynı embedding modeli
PARTI_BOYU = 16                         # tek HTTP isteğinde gömülecek metin sayısı (batch)


def metinleri_gom(metinler: list[str]) -> np.ndarray:
    """Metin listesini embedding matrisine çevirir (her satır bir metnin vektörü)."""
    vektorler = []
    for i in range(0, len(metinler), PARTI_BOYU):
        parti = metinler[i:i + PARTI_BOYU]
        yanit = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODELI, "input": parti},
            timeout=300,
        )
        yanit.raise_for_status()
        vektorler += yanit.json()["embeddings"]
    return np.array(vektorler, dtype=np.float32)


def kosinus_benzerligi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kosinüs benzerliği (cosine similarity): iki vektör arasındaki açının
    kosinüsü. 1'e yakın = anlamca yakın, 0'a yakın = alakasız."""
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a_norm @ b_norm.T


if __name__ == "__main__":
    # Anlamsal yakınlığı kelime eşleşmesi OLMADAN görelim:
    cumleler = [
        "Araç plakası nasıl sorgulanır?",
        "Taşıt kimliği sorgulama adımları",
        "Kar küreme ekipmanlarının bakımı",
        "Yemekhanede bugün ne var?",
    ]
    vektorler = metinleri_gom(cumleler)
    print(f"Model: {EMBED_MODELI} — vektör boyutu: {vektorler.shape[1]}\n")

    benzerlik = kosinus_benzerligi(vektorler, vektorler)
    for i, c in enumerate(cumleler):
        print(c)
        for j, d in enumerate(cumleler):
            if i != j:
                print(f"   {benzerlik[i, j]:.3f}  {d}")
        print()
    print("Gözlem: 1. ve 2. cümle ortak kelime içermediği halde en yüksek")
    print("benzerliği alır — dense retrieval'ın gücü tam olarak budur.")
