# Proba de viteză pe Jetson Nano

*Măsurat pe 6 noiembrie 2026.*

## Decizia

| Valoare | Alegere |
|---|---|
| Backend | `ort-gpu`, adică `onnxruntime_gpu` 1.11 cu `CUDAExecutionProvider` |
| `IMAGE_SIZE` | 256 |
| `USE_AUTOENCODER` | `True`, deci Task 7 din etapa 1 nu se activează |

Modelul complet durează 411 ms pe imagine, față de pragul de 800 ms și ținta de o secundă.

## Măsurătorile

Model neantrenat, cu 30 de imagini măsurate, după 5 de încălzire. Timp median, în milisecunde.

| Model | `ort-cpu` | `ort-gpu` | `ort-trt` | `tensorrt` |
|---|---:|---:|---:|---:|
| complet, 256 | 6135 | **411** | 399 | 393 |
| fără autoencoder, 256 | 4756 | 331 | 316 | 317 |
| fără autoencoder, 192 | 2502 | 183 | 169 | 170 |
| fără autoencoder, 160 | 1663 | 118 | 115 | 115 |

Memoria folosită în tot sistemul, în MB, cu modelul încărcat. Limita e 3000.

| Model | `ort-cpu` | `ort-gpu` | `ort-trt` | `tensorrt` |
|---|---:|---:|---:|---:|
| complet, 256 | 1930 | **2908** | 3109 | 2607 |
| fără autoencoder, 256 | 1683 | 2740 | 3010 | 2590 |
| fără autoencoder, 192 | 1592 | 2595 | 2919 | 2564 |
| fără autoencoder, 160 | 1496 | 2468 | 2853 | 2477 |

Timpul de încărcare pentru modelul complet: `ort-cpu` 0,4 s, `ort-gpu` 3,6 s, `ort-trt` 119 s,
`tensorrt` 305 s.

Valorile complete, cu timpul fiecărei imagini, sînt în `proba-viteza-rezultate.jsonl`.

## Condițiile

- Jetson Nano de 4 GB, JetPack 4.6, Python 3.6.9, TensorRT 8.2.1.8.
- Mod de putere MAXN, `jetson_clocks` activ, ventilator la maximum, car-logger oprit.
- `ort-cpu` a rulat cu `onnxruntime` 1.9, iar `tensorrt` cu `trtexec`, unealta NVIDIA, fiindcă
  TensorRT 8.2.1 n-are modulul `tensorrt.lite`.

## Verificările

- **Încălzirea:** combinația de control, modelul complet pe `ort-gpu`, a dat 411,3 ms la început și
  414,4 ms la final, o diferență de 0,8%, sub limita de 10%. Temperatura maximă a fost 61 °C.
- **Trecerea pe procesor:** nu e cazul. Placa video e de 13–15 ori mai rapidă decât procesorul, pe
  fiecare model.

## Cum s-a ales

Pe modelul complet, trei backend-uri încap în 800 ms, dar `ort-trt` depășește limita de memorie.
Rămân două.

Regula din plan alege cel mai rapid, adică `tensorrt`. S-a ales totuși `ort-gpu`, ca excepție:

| | `tensorrt` | `ort-gpu` |
|---|---|---|
| Timp | 393 ms | 411 ms, cu 4,6% mai mult |
| Memorie | 2607 MB | 2908 MB |
| Încărcare | 305 s, de redus cu un engine salvat | 3,6 s |
| Cod necesar | rulare nouă prin API-ul Python de nivel jos, nescrisă și nemăsurată | aceeași librărie ca pe laptop, cu cod aproape identic |

Câștigul de 18 ms nu schimbă nimic pentru țintă, iar `ort-gpu` e mai aproape de spec, §3: testele de
pe laptop spun ceva și despre Nano.

## Ce rămâne de urmărit

- **Memoria e strânsă.** `ort-gpu` a citit 2908 MB la prima rulare și 2624 MB la control, deci
  citirile variază cu aproape 300 MB. Sistemul singur ocupă cam 1,5–1,7 GB. Peste asta se vor adăuga
  camera și serverul web. Etapa 3 măsoară memoria pe aplicația completă. Dacă depășește limita,
  `tensorrt` e prima variantă de rezervă.
- **Timpul complet.** Cei 411 ms sînt doar modelul. Etapa 3 măsoară tot drumul până la verdict.
- **Setările.** Cifrele cer modul MAXN, `jetson_clocks` și ventilatorul pornit. `jetson_clocks` și
  ventilatorul nu rămân setate după o repornire, deci etapa 3 trebuie să le pornească.
- **Durata.** O rulare a durat sub un minut pe placa video. Încălzirea pe termen lung se vede abia la
  măsurătoarea de 10 minute din etapa 3.

## Varianta de rezervă

Dacă etapa 3 iese peste țintă, se propune modelul fără autoencoder: 331 ms la 256 sau 183 ms la 192,
pe `ort-gpu`. Scoaterea autoencoder-ului se face doar cu acordul lui Tudor.
