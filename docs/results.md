# Rezultate pe MVTec AD

*Măsurat pe 6 noiembrie 2026.*

## `metal_nut`, EfficientAD small, 256, cu autoencoder, 70.000 de pași

Scorul e AUROC la nivel de imagine: dintr-o piesă bună și una defectă, cât de des primește
cea defectă scorul mai mare.

| Măsură | Valoare |
|---|---:|
| AUROC, acest model, o singură antrenare | **0,9868** |
| Publicat, articolul EfficientAD, medie pe 5 rulări | 0,9949 |
| Publicat, `nelson1425/EfficientAD`, `results/mvtec_ad_small.json` | 0,9951 |

Modelul e sub scorurile publicate cu aproximativ 0,8 puncte procentuale și peste minimul de
0,90 al planului.

| Tip de defect | AUROC |
|---|---:|
| `bent` | 0,9927 |
| `color` | 0,9897 |
| `flip` | 1,0000 |
| `scratch` | 0,9644 |

## La pragul propus

Pragul propus e **0,3334**, calculat doar din cele 22 de imagini de validare: scorul lor cel
mai mare, plus 10%. Încă nu e confirmat.

| Imagini de test | Total | Marcate ca defecte |
|---|---:|---:|
| bune | 22 | 1, adică 4,5% |
| defecte | 93 | 91, adică 97,8% |

- Imaginea bună marcată greșit e `test/good/016.png`, cu scorul 0,917. Următoarea imagine bună
  are 0,261. Din cele 27 de perechi bună-defectă ordonate greșit, din 2046, 21 vin de la ea. Fără
  ea, AUROC ar fi 0,9969. Calculul arată cât cântărește o imagine, iar scorul raportat rămâne 0,9868.
- Defectele ratate sînt `test/scratch/000.png`, cu 0,299, și `test/scratch/020.png`, cu 0,167.

## Cum s-a verificat

Scorurile au fost recalculate pe laptop, din `model.onnx` descărcat, cu `onnxruntime` pe
procesor, nu cu codul de antrenare. AUROC, ratele și scorul pe fiecare tip de defect ies
identice cu cele din Colab. Pragul diferă cu 0,0000004.

## Limite

- **Datele:** site-ul MVTec nu răspundea pe 6 noiembrie 2026, așa că setul vine dintr-o copie
  neoficială: Hugging Face `MSherbinii/mvtec-ad-metal-nut`, revizia
  `8db7371c35a94709cbb3e69ac69f80d2716e2f8a`. Copia nu a fost comparată încă cu originalul.
- **Comparația:** e o singură antrenare, față de o medie pe 5 rulări, cu 22 de imagini bune
  în test. Cauza diferenței nu e stabilită.
- **Pregătirea imaginii:** aici se folosește OpenCV `INTER_AREA`, iar scorul e luat direct din
  harta de 56 x 56. Sursa folosește redimensionarea din PIL și ia scorul din harta mărită la
  rezoluția imaginii. Efectul acestor diferențe nu a fost măsurat.
- **Ce nu arată:** e un rezultat pe un set de referință, cu imagini de laborator, nu pe
  obiectele și camera stației. Timpul și memoria pe Nano sînt în `proba-viteza.md`.

## Surse

- [EfficientAD, articolul](https://arxiv.org/abs/2303.14535), fișierul
  [`per_scenario_results.json`](https://www.mydrive.ch/shares/79401/c41dcdb937972fb43d5cdd7bfa7072f8/download/449203483-1690527455/per_scenario_results.json),
  `efficientad_s/mvtec_ad/detection/auc_roc/metal_nut`.
- [`nelson1425/EfficientAD`](https://github.com/nelson1425/EfficientAD),
  `results/mvtec_ad_small.json`, `metal_nut/classification_au_roc`.
