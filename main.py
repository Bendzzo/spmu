from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom


priecinok = Path(__file__).resolve().parent / "DICOM images-20261006"
subory = sorted(priecinok.glob("*.dcm"))

if not subory:
    raise ValueError("Ziadne DICOM subory.")

stlpce = min(4, len(subory))
riadky = (len(subory) + stlpce - 1) // stlpce

fig, osi = plt.subplots(
    riadky,
    stlpce,
    figsize=(4 * stlpce, 4 * riadky),
    squeeze=False
)

for cesta, os in zip(subory, osi.flat):
    ds = pydicom.dcmread(cesta)
    obraz = ds.pixel_array

    if obraz.ndim != 2:
        raise ValueError(f"{cesta.name}: obraz nie je 2D.")

    print(
        f"{cesta.name}: "
        f"rozmery={obraz.shape}, "
        f"min={np.min(obraz)}, "
        f"max={np.max(obraz)}"
    )

    farebna_mapa = (
        "gray_r"
        if ds.get("PhotometricInterpretation") == "MONOCHROME1"
        else "gray"
    )

    os.imshow(obraz, cmap=farebna_mapa)
    os.set_title(f"{cesta.name}\n{obraz.shape[0]} × {obraz.shape[1]}")
    os.axis("off")

for os in osi.flat:
    os.axis("off")

fig.tight_layout()
plt.show()