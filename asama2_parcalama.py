"""Aşama 2 — Parçalama (Chunking) ve Parça Metadata'sı.

Belgeyi olduğu gibi modele veremeyiz:
1. Embedding modelleri uzun metinde anlamı "sulandırır" — küçük parçalar
   daha isabetli aranır.
2. LLM bağlam penceresi (context window) sınırlıdır; 100 sayfayı değil,
   en alakalı 4-5 parçayı göndeririz.

İki ayar her RAG projesinin klasik tartışmasıdır:
- PARCA_BOYU (chunk size): parça küçükse bağlam kopar, büyükse arama bulanıklaşır.
- BINDIRME (chunk overlap): ardışık parçaların ortak kısmı; cümlenin
  parça sınırında ikiye bölünüp anlamsızlaşmasını yumuşatır.

Chunking metadata'nın alt konusu DEĞİLDİR — kardeş kavramlardır:
parçalama bittikten sonra her parçaya kaynağını söyleyen metadata eklenir
(chunk metadata). Sitasyon bu metadata'dan üretilir.

Çalıştır:  python asama2_parcalama.py
"""

from asama1_belge_okuma import belgeleri_oku

PARCA_BOYU = 800   # hedef parça uzunluğu, karakter (chunk size)
BINDIRME = 150     # ardışık parçaların ortak kısmı, karakter (chunk overlap)


def parcala(metin: str, boy: int = PARCA_BOYU, bindirme: int = BINDIRME) -> list[str]:
    """Metni yaklaşık `boy` karakterlik, `bindirme` kadar örtüşen parçalara böler.

    Naive strateji: kayan pencere + kelimeyi ortadan bölmemek için pencere
    içindeki son boşluğa geri çekilme. Recursive / semantic / başlık-bazlı
    chunking stratejileri geliştirme ödevidir.
    """
    metin = metin.strip()
    if len(metin) <= boy:
        return [metin] if metin else []

    parcalar = []
    baslangic = 0
    while baslangic < len(metin):
        son = min(baslangic + boy, len(metin))
        if son < len(metin):
            bosluk = metin.rfind(" ", baslangic, son)
            if bosluk > baslangic:
                son = bosluk
        parcalar.append(metin[baslangic:son].strip())
        if son >= len(metin):
            break
        sonraki = son - bindirme
        bosluk = metin.rfind(" ", baslangic, sonraki)
        if bosluk > baslangic:  # bindirme başlangıcını kelime sınırına çek
            sonraki = bosluk + 1
        if sonraki <= baslangic:  # güvenlik: her turda ilerleme garantisi
            sonraki = son
        baslangic = sonraki
    return [p for p in parcalar if p]


def belgeleri_parcala(kayitlar: list[dict]) -> list[dict]:
    """Her kaydı parçalara böler ve her parçaya metadata ekler.

    Parça metadata'sı (chunk metadata), retrieval sonucunu kaynağına
    bağlar — doğru sitasyonun ön koşulu budur.
    """
    parcalar = []
    for kayit in kayitlar:
        for i, metin in enumerate(parcala(kayit["metin"]), start=1):
            parcalar.append({
                "dosya": kayit["dosya"],
                "sayfa": kayit["sayfa"],
                "parca_no": i,
                "metin": metin,
            })
    return parcalar


def kaynak_etiketi(parca: dict) -> str:
    """Sitasyon etiketi: [dosya, sayfa 3] ya da sayfasız dosyada [dosya, parça 2]."""
    if parca["sayfa"]:
        return f"[{parca['dosya']}, sayfa {parca['sayfa']}]"
    return f"[{parca['dosya']}, parça {parca['parca_no']}]"


if __name__ == "__main__":
    parcalar = belgeleri_parcala(belgeleri_oku())
    boylar = [len(p["metin"]) for p in parcalar]
    print(f"{len(parcalar)} parça üretildi "
          f"(ortalama {sum(boylar) // len(boylar)} karakter, "
          f"en küçük {min(boylar)}, en büyük {max(boylar)})\n")

    for p in parcalar[:2]:
        print(f"{kaynak_etiketi(p)}")
        print(f"  {p['metin'][:120]}...\n")

    # Bindirmeyi (overlap) gözle görelim: 1. parçanın sonu ile 2. parçanın başı
    ayni_dosya = [p for p in parcalar if p["dosya"] == parcalar[0]["dosya"]]
    if len(ayni_dosya) >= 2:
        print("Bindirme örneği — 1. parçanın SONU:")
        print(f"  ...{ayni_dosya[0]['metin'][-100:]}")
        print("2. parçanın BAŞI:")
        print(f"  {ayni_dosya[1]['metin'][:100]}...")
