# Porovnanie detekcií

Model: YOLO26s, vstup 640 px, CPU, NMS IoU = 0.7.

| Obrázok | Prah 0.25 | Prah 0.90 |
| --- | --- | --- |
| bus.jpg | 5 | 2 |
| zidane.jpg | 3 | 1 |
| dog.jpg | 1 | 1 |

Nižší prah ponecháva aj menej isté predikcie; medzi nimi môžu byť správne aj chybné detekcie. Vyšší prah filtruje menej isté rámčeky, ale môže vynechať skutočné objekty. Skóre istoty nie je nameraná presnosť modelu; na tú treba referenčné anotácie.

Konkrétna vizuálne overená chyba a jej možné príčiny sú opísané v ../POZOROVANIA.md.
