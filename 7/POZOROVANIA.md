# Pozorovania k cvičeniu 7

## Model a vstupy (úlohy 1 až 3)

Použil som predtrénovaný model YOLO26s s 80 triedami datasetu COCO. Rozpoznáva
napríklad osoby, autá, autobusy, psy, mačky, kravaty a dopravné značky; celý zoznam
je v `vysledky/triedy.txt`. Na troch rôznych fotografiách som vykonal detekciu
pri vstupnej veľkosti 640 px na CPU. Výsledné fotografie obsahujú ohraničujúce
rámčeky, názvy tried a skóre istoty. Pri porovnaní sa menil iba prah istoty;
NMS IoU zostalo 0.7. Používa sa predvolená vetva s NMS.

## Porovnanie prahov (úloha 4)

| Fotografia | Prah 0.25 | Prah 0.90 |
| --- | --- | --- |
| bus.jpg | 5: autobus a 4 osoby | 2: autobus a 1 osoba |
| zidane.jpg | 3: 2 osoby a 1 kravata | 1: osoba vpravo |
| dog.jpg | 1: pes | 1: pes |

Na fotografii autobusu sa pri vyššom prahu stratili tri osoby so skóre približne
0.898, 0.846 a 0.833. Zostal autobus (0.923) a osoba na pravom okraji (0.923).
Pri fotografii dvoch ľudí zostala pravá osoba (0.928), no zmizla ľavá osoba
(0.897) a jej správne nájdená kravata (0.867). Pes mal skóre 0.921, preto zostal
pri oboch prahoch. Presné hodnoty a súradnice sú v `vysledky/detekcie.json`.

Najprv som vyskúšal prahy 0.25 a 0.75. Na týchto troch fotografiách sa počty
ani triedy nezmenili, pretože všetky zachované detekcie mali skóre nad 0.75.
Pre názorné finálne porovnanie som druhý prah zvýšil na 0.90.

## Konkrétna chyba a možné príčiny (úloha 5)

Na `zidane.jpg` je viditeľná aj kravata pravého muža, no model jej pri prahu
0.25 nepriradil samostatný rámček. Na obrázku sú teda dve viditeľné kravaty,
ale model našiel iba kravatu ľavého muža. Ide o vynechaný objekt z podporovanej
triedy `tie`. Kravata pravého muža je čiastočne zakrytá rukou a má malú plochu;
prekrytie, vzhľad a zmenšenie vstupu mohli detekciu sťažiť. Sú to možné príčiny,
nie overené vysvetlenie vnútorného rozhodovania siete.

Ďalšie vynechania vznikli pri prahu 0.90: skutočné osoby a ľavá kravata zmizli,
pretože ich skóre bolo pod nastaveným prahom. Táto príčina je priamo doložená
skóre detekcií pri nižšom prahu.

## Zhrnutie (úloha 6)

YOLO26s na fotografii psa vytvoril jednu správnu detekciu a nepridal chybnú
triedu `cat`, ktorú na tomto obrázku vytvoril predchádzajúci YOLOv8n. Zlepšenie
na týchto ukážkach nepredstavuje meranie celkovej presnosti modelov.
Vyšší prah 0.90 odstránil niekoľko správnych detekcií osôb a kravaty, pričom
pes zostal zachovaný. Vysoký prah teda nezaručuje lepší výsledok pre každú úlohu.
Skóre istoty nemožno zamieňať s nameranou presnosťou; bez referenčných anotácií
nevyhodnocujem precision, recall ani mAP. Šesť anotovaných fotografií a spoločné
porovnanie sú uložené vo `vysledky/`.

Pozorovania opisujú skutočný beh s dodanými obrázkami a modelom YOLO26s
(Ultralytics 8.4.211). Výsledky som overil aj vizuálne.
