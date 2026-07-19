# Naive RAG — Stajyer Eğitim Projesi

Kurumsal RAG'e giden yolun ilk adımı: her aşaması **tek başına çalıştırılabilir**,
framework'süz (LlamaIndex/LangChain yok), tamamen lokal bir Naive RAG.

Amaç kodu ezberlemek değil, **her katmanın ne işe yaradığını görmek**:
kütüphane sihri olmadığı için chunking, embedding, retrieval ve prompt'un
tamamı gözünüzün önünde.

## Büyük resim

```
Belgeler (dokumanlar/)
      │
      ▼
[Aşama 1] Belge Okuma (Parsing)
      │
      ▼
[Aşama 2] Parçalama (Chunking) + Metadata
      │
      ▼
[Aşama 3] Gömme (Embedding — BGE-M3)
      │
      ▼
[Aşama 4] Vektör Deposu (Vector Store — kalıcı)
      │
      ▼
[Aşama 5] Getirme (Retrieval — dense, top-k, eşik)
      │
      ▼
[Aşama 6] Prompt + Lokal LLM (qwen3:4b) → KAYNAKLI yanıt
```

## Kurulum

1. [Ollama](https://ollama.com) kurun ve modelleri indirin (tek seferlik, internet gerekir):

   ```
   ollama pull bge-m3      # embedding modeli (kurumsal projeyle aynı)
   ollama pull qwen3:4b    # yanıt üreten lokal LLM
   ```

2. Python ortamı (3.10+):

   ```powershell
   # Windows
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```

   ```bash
   # macOS / Linux
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

## Aşamalar

Sırayla çalıştırın; her dosya bir öncekinin fonksiyonlarını import eder
(modülerliğin en yalın hali). Her dosyanın başındaki docstring o aşamanın
"ders notu"dur — **önce okuyun, sonra çalıştırın, sonra kurcalayın.**

| Aşama | Dosya | Eğitim konusu | Öğrenilen kavramlar |
|---|---|---|---|
| 1 | `asama1_belge_okuma.py` | Belge Hazırlama (Ingestion) | parsing, sayfa metadata'sı |
| 2 | `asama2_parcalama.py` | Chunking ve Metadata Yönetimi | chunk size, overlap, chunk metadata, sitasyon etiketi |
| 3 | `asama3_gomme.py` | Embedding | vektör, kosinüs benzerliği (cosine similarity) |
| 4 | `asama4_vektor_deposu.py` | Vector Database | indeksleme, kalıcılık (persistence) |
| 5 | `asama5_getirme.py` | Retrieval | dense search, top-k, benzerlik eşiği |
| 6 | `asama6_naive_rag.py` | Prompting + Local LLM | bağlam prompt'u, kaynak gösterme (citation), halüsinasyon önleme |

```
python asama1_belge_okuma.py
python asama2_parcalama.py
python asama3_gomme.py        # buradan itibaren Ollama çalışıyor olmalı
python asama4_vektor_deposu.py
python asama5_getirme.py      # retrieval'ı LLM'siz test edin — önemli alışkanlık!
python asama6_naive_rag.py    # uçtan uca sohbet
```

Deneme soruları (örnek dokümanlara göre):

- "HGS ile SGS arasındaki fark nedir?"
- "Taşlıdere Köprüsünün yük sınırı kaç ton?"  ← PDF içindeki tablodan gelir.
- "5. Bölgede kaç kar küreme aracı var?"  ← Excel'den gelir; sitasyonda sheet adı görünür.
- "Kar mücadelesinde hangi araçlar kullanılıyor?"  ← dokümanda "kar küreme
  ekipmanları" yazar; dense retrieval'ın kelime eşleşmesi olmadan bulduğunu görün.
- "Yemekhane kaçta açılıyor?" ← dokümanlarda YOK; sistemin "bilmiyorum"
  demesi gerekir (halüsinasyon testi).

## Test korpusu ve değerlendirme

`dokumanlar/` klasöründe kurgusal bir karayolları kurumuna ait **29 dosyalık**
test korpusu bulunur: 9 txt, 8 docx, 7 pdf (tablolu), 5 xlsx (çok sayfalı).
Konular: ücretlendirme (HGS/SGS), bakım ve kar mücadelesi, tünel/köprü
güvenliği, denetim istasyonları (WIM), akıllı ulaşım sistemleri, hizmet
tesisleri, İSG, çevre, ihale-hakediş ve istatistik tabloları. Belgeler
birbirine çapraz atıf yapar; sayılar dosyalar arasında tutarlıdır.
Tüm içerik kurgusaldır.

`degerlendirme/sorular.csv` bu korpus için hazırlanmış **132 soruluk** settir:
her satırda soru, beklenen cevap, kaynak dosya(lar) ve soru tipi
(`tekil`, `sayisal`, `es_anlamli`, `coklu_dokuman`, `yok`) bulunur.
`yok` tipi soruların cevabı korpusta bilerek YOKTUR — halüsinasyon testi.

```
python degerlendirme.py            # retrieval isabet oranını ölç (indeks kurulu olmalı)
python degerlendirme.py --ayrinti  # başarısız soruları listele
python degerlendirme.py --k 8 --esik 0.5   # parametre deneyleri
```

Altın kural: **her geliştirmeden önce ve sonra bu scripti çalıştırın**;
"daha iyi oldu" hissi değil, isabet oranındaki değişim konuşur.

Kendi dosyalarınızı da `dokumanlar/` klasörüne atıp `asama4`'ü yeniden
çalıştırabilirsiniz (taranmış/görüntü PDF'ler OCR olmadığı için boş çıkar).

## Analiz soruları (sunum öncesi cevaplayın)

1. Neden sadece embedding araması yetmez? BM25 ile dense search birbirini nasıl tamamlar?
2. Top-K kaç seçilmeli? Performans ve doğruluğa etkisi ne?
3. Chunk boyutu büyüyünce/küçülünce retrieval nasıl değişiyor? (Deneyin!)
4. Benzerlik eşiği (`ESIK`) neden var? Kaldırınca ne oluyor?
5. Metadata filtreleme ne zaman gerekli olur?
6. Sitasyon neden retrieval katmanında hazırlanır, LLM'e neden güvenilmez?

## Sonrası

Bu naive sürümün **bilinçli eksikleri** var. [gelistirme_odevleri.md](gelistirme_odevleri.md)
dosyası her eksiği bir ödeve dönüştürür (hybrid search, RRF, Chroma, artımlı
indeksleme, değerlendirme...). Konuyu anlayıp analiz ettikten sonra
geliştirmeleri bu liste üzerinden yapacağız.
