from pathlib import Path

import pydicom
import csv

POLOZKY = (
    "PatientSex",
    "PatientAge",
    "PatientPosition",
    "Modality",
    "Manufacturer",
    "StudyDescription",
    "ProtocolName",
)

def nacitaj_metadata(cesty):
    vysledky = []

    for cesta in cesty:
        ds = pydicom.dcmread(cesta, stop_before_pixels=True)
        metadata = {}

        for polozka in POLOZKY:
            hodnota = ds.get(polozka, None)

            if hodnota is None:
                metadata[polozka] = None
            else:
                text = str(hodnota).strip()
                metadata[polozka] = text if text else None

        vysledky.append(metadata)

    return vysledky


priecinok = Path(__file__).resolve().parent / "DICOM images-20261006"
subory = sorted(priecinok.glob("*.dcm"))

if not subory:
    raise ValueError("Ziadne DICOM subory.")

vysledky = nacitaj_metadata(subory)

for cesta, metadata in zip(subory, vysledky):
    print(f"\n{cesta.name}")

    for polozka, hodnota in metadata.items():
        text = hodnota if hodnota is not None else "-"
        print(f"  {polozka}: {text}")

print("\nPorovnanie:")

for polozka in POLOZKY:
    hodnoty = {metadata[polozka] for metadata in vysledky}

    if len(hodnoty) == 1:
        print(f"{polozka}: rovnake")
    else:
        print(f"{polozka}: rozne ({len(hodnoty)})")

dalsie_polozky = (
    "SOPInstanceUID",
    "InstanceNumber",
    "ImagePositionPatient",
)

print("\nDalsie udaje:")

for polozka in dalsie_polozky:
    hodnoty = set()

    for cesta in subory:
        ds = pydicom.dcmread(cesta, stop_before_pixels=True)
        hodnota = ds.get(polozka, None)
        text = str(hodnota).strip() if hodnota is not None else ""
        hodnoty.add(text if text else None)

    stav = "rovnake" if len(hodnoty) == 1 else "rozne"
    print(f"{polozka}: {stav}, pocet hodnot={len(hodnoty)}")

vystup = Path(__file__).resolve().parent / "metadata.csv"

with vystup.open("w", newline="", encoding="utf-8-sig") as subor:
    zapisovac = csv.DictWriter(
        subor,
        fieldnames=["Subor", *POLOZKY],
        delimiter=";"
    )

    zapisovac.writeheader()

    for cesta, metadata in zip(subory, vysledky):
        riadok = {"Subor": cesta.name, **metadata}
        zapisovac.writerow(riadok)

print(f"\nUlozene: {vystup.name}")