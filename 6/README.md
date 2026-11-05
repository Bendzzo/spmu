# Cvičenie 6 – analýza EEG

Riešenie všetkých šiestich úloh je v `main.py`. Používa iba NumPy a Matplotlib.

## Spustenie

Z koreňového priečinka `cvika` v PowerShelli, s pripraveným lokálnym prostredím:

```powershell
& './6/.venv/Scripts/python.exe' -X utf8 6/main.py
```

Program uloží výsledky do `6/vystupy` a otvorí graf. Ak nechceš otvárať okno:

```powershell
& './6/.venv/Scripts/python.exe' -X utf8 6/main.py --no-show
```

Na inom počítači stačí nainštalovať závislosti do vlastného Python prostredia:

```powershell
python -m pip install -r 6/requirements.txt
python -X utf8 6/main.py
```

Cesty k vstupom a výstupom sa predvolene odvodzujú od polohy `main.py`, preto
program funguje aj pri spustení z iného pracovného priečinka. Voliteľné argumenty
`--data-dir` a `--output-dir` umožňujú použiť inú kópiu rovnakého datasetu Bonn
alebo iný výstupný priečinok.

## 1. Príprava a význam skupín

| Priečinok | Sada Bonn | Význam | Počet meraní | Riadky matice od nuly |
| --- | --- | --- | ---: | --- |
| Z | A | Zdraví dobrovoľníci, otvorené oči, povrchové EEG | 100 | 0–99 |
| O | B | Zdraví dobrovoľníci, zatvorené oči, povrchové EEG | 100 | 100–199 |
| N | C | Pacienti mimo záchvatu, hipokampus opačnej hemisféry | 100 | 200–299 |
| F | D | Pacienti mimo záchvatu, epileptogénna oblasť | 100 | 300–399 |
| S | E | Pacienti počas epileptického záchvatu | 100 | 400–499 |

Záchvatové merania obsahuje skupina **S**. Skupiny N a F pochádzajú od pacientov
s epilepsiou, ale samotné tieto úseky boli zaznamenané mimo záchvatu.
Skupiny N, F a S sú intrakraniálne záznamy.

Význam skupín vychádza z pôvodnej publikácie (časť II.A):
[Andrzejak a kol. (2001), Physical Review E 64, 061907](https://www.upf.edu/documents/229517819/232450661/Andrzejak-PhysicalReviewE2001.pdf/0e9a54b8-8993-b400-743e-4d64fa29fb63).
Priradenie priečinkov k sadám a vzorkovacia frekvencia **173,61 Hz** sú uvedené na
[stránke autorov datasetu](https://www.upf.edu/web/ntsa/downloads/-/asset_publisher/xvT6E4pczrBw/content/2001-indications-of-nonlinear-deterministic-and-finite-dimensional-structures-in-time-series-of-brain-electrical-activity-dependence-on-recording-regi).

## 2. Načítanie a kontrola

Každý TXT súbor obsahuje jeden kanál EEG, jednu číselnú vzorku na riadok, bez
hlavičky. `load_measurement()` odmietne prázdny súbor, prázdny riadok, viac
stĺpcov, nečíselné hodnoty, NaN aj nekonečno. Pri načítaní sa nerozlišuje veľkosť
písmen prípony, takže sa spracujú aj súbory `.TXT` zo skupiny N.

V dodaných dátach má **všetkých 500 meraní presne 4097 vzoriek**, bez chýbajúcich
hodnôt. Všetky hodnoty sú celočíselné; v matici sú uložené ako `float64`, aby sa
s nimi dalo ďalej numericky pracovať. Dokumentačná stránka uvádza 4096 vzoriek,
ale riešenie používa skutočný počet z lokálnych súborov a žiadnu vzorku neodstraňuje.

Ak by sa dĺžky líšili, program vypíše ich počty a skončí s vysvetlením. Možné
riešenie by bolo zvoliť spoločný časový úsek a všetky merania orezať na dĺžku
najkratšieho. Pri rovnakej frekvencii by sa tým zachovala rovnaká dĺžka trvania,
ale stratili by sa konce dlhších meraní. Dopĺňanie nulami by zas vytváralo umelé
časti signálu. Pre tieto dáta netreba použiť ani jeden postup.

## 3. Spoločná matica

`load_dataset()` postupne načíta skupiny v poradí `Z, O, N, F, S`; súbory v každej
skupine zoradí podľa názvu. Pomocou `np.stack()` uloží merania po riadkoch do
matice **(500, 4097)**. Ku každému riadku uchová skupinu, názov súboru a príznak
záchvatu. Posledných 100 riadkov teda obsahuje záchvatové merania.

## 4. Uloženie a opätovné načítanie

Komprimovaný archív `vystupy/eeg_merania.npz` obsahuje:

| Kľúč | Tvar | Obsah |
| --- | --- | --- |
| `signals` | `(500, 4097)` | Matica pôvodných vzoriek |
| `groups` | `(500,)` | Zdrojová skupina každého riadka |
| `filenames` | `(500,)` | Pôvodný názov súboru |
| `is_seizure` | `(500,)` | Boolean, True iba pre skupinu S |
| `sampling_rate_hz` | skalár | Vzorkovacia frekvencia 173,61 Hz |

`save_and_verify()` archív hneď opäť načíta cez `np.load(..., allow_pickle=False)`
a skontroluje názvy polí, dátové typy aj presnú rovnosť všetkých hodnôt.
Na ďalšom cvičení ho môžeš použiť takto:

```python
from pathlib import Path
import numpy as np

archive = Path(__file__).resolve().parent / "vystupy" / "eeg_merania.npz"
with np.load(archive, allow_pickle=False) as data:
    signals = data["signals"]
    groups = data["groups"]
    filenames = data["filenames"]
    is_seizure = data["is_seizure"]
    sampling_rate_hz = float(data["sampling_rate_hz"])
```

## 5. Časové priebehy

`plot_measurements()` vyberie prvý súbor každej skupiny. Vľavo zobrazí celý
záznam, vpravo prvých 5 sekúnd. Čas vzorky sa vypočíta ako `index / fs`.
Trvanie `N/fs` je približne **23,599 s**; posledná vzorka má čas `(N-1)/fs`.
Každý panel má označenú skupinu, os času a amplitúdy. Všetkých desať panelov
používa rovnaké limity amplitúdy. Skupina S je zvýraznená červenou.

Graf sa uloží ako `vystupy/porovnanie_eeg.png`. Amplitúda je označená ako pôvodná
hodnota zo súboru; program nevykonáva nezdokumentovaný prevod jednotiek,
normalizáciu ani filtráciu.

## 6. Vizuálne porovnanie a pozorovania

V ukážke Z001 sa signál nepravidelne mení okolo nuly a má relatívne malé
amplitúdy. V O001 sú miestami viditeľné pravidelnejšie krátke úseky kmitania;
rozsah hodnôt je −164 až 225, kým Z001 má −190 až 185. N001 obsahuje aj
izolované ostrejšie záporné výkyvy. F001 je v tomto výbere pokojnejší a má
najmenšiu smerodajnú odchýlku, približne 28,63.

S001 sa na spoločnej mierke odlišuje veľkými výkyvmi **−1765 až 1027**
a opakovanými ostrými zápornými poklesmi. Tieto poklesy tvoria opakujúci sa
motív, hoci ich rozstupy aj veľkosť kolíšu; nejde o dokonale periodický signál.
Rozpätie S001 je 2792 oproti 375 v Z001. Smerodajná odchýlka S001 je približne
478,48, zatiaľ čo pri ostatných štyroch ukážkach je 28,63 až 50,53.

Aj pri všetkých meraniach má S najväčší medián rozpätia max–min: **1729**,
oproti 307,5 v Z, 412 v O, 365,5 v N a 370 v F. Zároveň sa v celej skupine F
vyskytujú aj veľké extrémy, takže pokojný F001 nereprezentuje všetky jej
merania. Pri interpretácii rozdielov treba zohľadniť aj odlišné umiestnenie
elektród pri povrchovom a intrakraniálnom EEG.

Presné počty, rozsahy a kontrolné výsledky program pri každom spustení vypíše
a uloží do `vystupy/kontrola_dat.txt`.
