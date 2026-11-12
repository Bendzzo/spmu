# Cvičenie 7 - detekcia objektov

Riešenie používa predtrénovaný YOLO26s bez ďalšieho trénovania. Spracuje tri
fotografie (autobus s chodcami, dvaja ľudia s kravatami a pes) pri prahoch istoty
0.25 a 0.90. Komentáre v `main.py` označujú hlavné časti programu.

## Spustenie v PowerShelli z koreňa projektu

```powershell
python -m venv 7/.venv
& ./7/.venv/Scripts/python.exe -m pip install -r 7/requirements.txt
& ./7/.venv/Scripts/python.exe 7/main.py
```

Pre spustenie bez grafického okna pridaj `--no-show`. V IDE nastav interpreter
`7/.venv/Scripts/python.exe`. Cesty sú odvodené od umiestnenia skriptu, takže
pracovný priečinok pri spustení nerozhoduje. Prvý beh potrebuje internet na
stiahnutie váh a obrázkov; ďalšie behy používajú lokálne súbory.

## Výstupy

- `vysledky/triedy.txt`: všetky triedy podporované modelom.
- `vysledky/*_conf_*.jpg`: šesť fotografií s rámčekmi, triedami a istotou.
- `vysledky/porovnanie.png`: spoločné porovnanie troch obrázkov a oboch prahov.
- `vysledky/pocty.csv`, `vysledky/detekcie.json`: počty a podrobné detekcie.
- `vysledky/porovnanie.md`: zhrnutie počtov pre aktuálny beh.
- `POZOROVANIA.md`: rozbor výsledkov a konkrétnej chyby (úlohy 5 a 6).

## Zdroje

- Python API: https://docs.ultralytics.com/tasks/detect/
- Váhy: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
- Autobus: https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg
- Ľudia: https://raw.githubusercontent.com/ultralytics/assets/main/im/zidane.jpg
- Pes: https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg

Fotografie sú verejné ukážkové vstupy uvedených projektov. Detektor neidentifikuje
konkrétne osoby, iba triedy objektov. Názvy tried zostávajú v angličtine podľa modelu.
