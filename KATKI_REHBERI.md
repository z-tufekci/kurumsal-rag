# Katkı Rehberi — Ödev Fazı İş Akışı

Bu repo gerçek bir ekip gibi çalışır: **her ödev bir issue, her issue bir
dal (branch), her dal bir PR ile kapanır.** `main` dalı korumalıdır —
doğrudan push edilemez; kod yalnızca incelenmiş PR ile girer.

## Ödev döngüsü (adım adım)

1. **Issue'nu üstlen:** Sana atanan issue'yu aç, hedefi ve kabul
   kriterlerini oku. Soru varsa issue'ya yorum yaz.
2. **Güncel main'den dal aç:**
   ```
   git checkout main && git pull
   git checkout -b odev2-bm25        # kalıp: odevN-kisa-ad
   ```
3. **Küçük commit'lerle çalış:** Her commit tek bir mantıklı adım olsun;
   mesaj Türkçe ve emir kipinde ("bm25 indeks kurulumu eklendi" değil,
   "bm25 indeksini kur").
4. **Deneylerini çalıştır:** `python degerlendirme.py` — **öncesi/sonrası
   sayıları not et.** Sayısız PR açma; "bence daha iyi" gerekçe değildir.
5. **PR aç:**
   ```
   git push -u origin odev2-bm25
   ```
   GitHub'da Pull Request oluştur; açıklamaya **`Closes #9`** yaz (issue
   numaran neyse) — merge olunca issue otomatik kapanır. Kabul kriterleri
   kutucuklarını doldur, öncesi/sonrası sayılarını tabloyla ekle.
6. **Çapraz inceleme:** PR'ına bir arkadaşını reviewer olarak ekle
   (Reviewers menüsü). En az 1 onay olmadan merge edilemez (kural
   GitHub'da zorunlu). Mentor da her PR'a bakar.
7. **Merge sonrası:** Dalını silebilirsin; `main`'e dönüp `git pull` yap.

## İnceleme (review) görgü kuralları

- Yorum **koda** yapılır, kişiye değil. "Bu neden böyle?" sorusu her zaman
  meşrudur; "olmamış" tek başına yorum değildir.
- İncelediğin PR'da en az bir şeyi çalıştırıp dene — okumak yetmez.
- Değişiklik iste (Request changes) korkma; erken yakalanan hata ucuzdur.

## Günlük ritim

- **Sabah:** `git checkout main && git pull` (arkadaşlarının merge'lerini al).
- **Gün sonu:** Kendi issue'na 3 satır yorum: *bugün ne yaptım / yarın ne
  yapacağım / takıldığım yer.* Mentor takibi buradan yürür; takıldığını
  saklamak zaman kaybettirir, yazmak kazandırır.
- **Cuma:** Sunum — problem → çözüm → sayılar ([SUNUM_REHBERI.pdf](SUNUM_REHBERI.pdf)).

## Sık yaşanan durumlar

- **`git pull` "local changes would be overwritten" hatası:** Yerel
  deneme değişikliğin var demektir. Ya at (`git restore <dosya>`) ya
  kenara al (`git stash`, pull, `git stash pop`). Deneylerini dal üzerinde
  yaparsan bu hatayı hiç görmezsin.
- **İndeks bozuldu / sonuçlar tuhaf:** `indeks/` klasörünü sil,
  `python asama4_vektor_deposu.py` ile yeniden kur. İndeks türetilmiş
  veridir, silmek güvenlidir (bu yüzden git'te de yoktur).
- **Aynı dosyaya iki kişi dokunuyor** (herkes her ödevi seçebildiği için
  bu sık olur, örn. birden fazla kişi `asama1`'e dokunabilir): küçük PR'lar
  açın, birbirinizi bekletmeyin; çakışma çıkarsa birlikte çözün. Bir ödevi
  seçmeden önce ilgili issue'ya "üzerinde çalışıyorum" diye yorum bırakmak
  mükerrer işi önler.

## Yapma listesi

- `main`'e doğrudan push deneme (engellenir, ama deneme de).
- `dokumanlar/` içine ödevle ilgisiz dosya koyma — oradaki her şey
  indekslenir ve herkesin retrieval sonuçlarını değiştirir.
- `.venv/` veya `indeks/` klasörlerini commit'leme (gitignore'da ama
  `git add -f` ile zorlamak mümkün — zorlamayın).
- Başkasının dalına ondan habersiz push etme.
