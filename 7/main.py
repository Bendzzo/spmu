"""Cvicenie 7: detekcia objektov pomocou YOLO26s."""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import matplotlib.pyplot as plt
import requests
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results


IMAGES = {
    "bus.jpg": "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg",
    "zidane.jpg": "https://raw.githubusercontent.com/ultralytics/assets/main/im/zidane.jpg",
    "dog.jpg": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
}
MODEL_NAME = "YOLO26s"
MODEL_FILE = "yolo26s.pt"
MODEL_URL = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{MODEL_FILE}"
THRESHOLDS = (0.25, 0.90)


def download(url, destination):
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Sťahujem {destination.name} ...")
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    destination.write_bytes(response.content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Iba uložiť výsledky, neotvárať okno.")
    args = parser.parse_args()
    output_dir = BASE_DIR / "vysledky"
    output_dir.mkdir(exist_ok=True)

    # Nacitanie modelu a zoznamu tried.
    model_path = BASE_DIR / "modely" / MODEL_FILE
    download(MODEL_URL, model_path)
    model = YOLO(str(model_path))
    classes = "\n".join(f"{class_id}: {name}" for class_id, name in model.names.items())
    print(f"\nModel {MODEL_NAME} rozpoznáva {len(model.names)} tried:\n{classes}\n")
    (output_dir / "triedy.txt").write_text(classes + "\n", encoding="utf-8")

    # Priprava obrazkov.
    paths = []
    for filename, url in IMAGES.items():
        path = BASE_DIR / "obrazky" / filename
        download(url, path)
        paths.append(path)

    # Detekcia a porovnanie prahov.
    figure, axes = plt.subplots(len(paths), len(THRESHOLDS), figsize=(14, 15))
    all_detections = []
    summaries = []
    for row, path in enumerate(paths):
        for col, threshold in enumerate(THRESHOLDS):
            predictions = model.predict(
                source=str(path), conf=threshold, iou=0.7, imgsz=640,
                device="cpu", verbose=False, stream=False,
            )
            result = next(iter(predictions), None)
            if not isinstance(result, Results):
                raise RuntimeError("Model nevrátil výsledok detekcie objektov.")
            boxes = result.boxes
            if boxes is None:
                raise RuntimeError("Model nevrátil detekčné rámčeky. Použi model na detekciu objektov.")
            counts = Counter()
            annotated = result.plot(labels=True, conf=True, line_width=2)[:, :, ::-1]
            target = output_dir / f"{path.stem}_conf_{threshold:.2f}.jpg"
            Image.fromarray(annotated).save(target, quality=95)

            for box in boxes:
                class_id = int(box.cls.item())
                name = model.names[class_id]
                counts[name] += 1
                all_detections.append({
                    "obrazok": path.name,
                    "prah": threshold,
                    "trieda_id": class_id,
                    "trieda": name,
                    "istota": round(float(box.conf.item()), 6),
                    "xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                })

            total = len(boxes)
            summaries.append({
                "obrazok": path.name, "prah": threshold,
                "pocet": total, "triedy": dict(counts),
            })
            print(f"{path.name}, prah {threshold:.2f}: {total} objektov, {dict(counts)}")
            axes[row, col].imshow(annotated)
            axes[row, col].set_title(f"{path.name} | prah {threshold:.2f} | počet: {total}")
            axes[row, col].axis("off")

    figure.suptitle(f"{MODEL_NAME}: porovnanie prahov istoty", fontsize=18)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_dir / "porovnanie.png", dpi=150)

    # Export vysledkov.
    (output_dir / "detekcie.json").write_text(
        json.dumps(all_detections, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    with (output_dir / "pocty.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["obrazok", "prah", "pocet", "triedy"])
        writer.writeheader()
        for summary in summaries:
            writer.writerow({**summary, "triedy": json.dumps(summary["triedy"], ensure_ascii=False)})

    observations = [
        "# Porovnanie detekcií", "",
        f"Model: {MODEL_NAME}, vstup 640 px, CPU, NMS IoU = 0.7.", "",
        f"| Obrázok | Prah {THRESHOLDS[0]:.2f} | Prah {THRESHOLDS[1]:.2f} |",
        "| --- | --- | --- |",
    ]
    for path in paths:
        low, high = [item for item in summaries if item["obrazok"] == path.name]
        observations.append(f"| {path.name} | {low['pocet']} | {high['pocet']} |")
    observations.extend([
        "", "Nižší prah ponecháva aj menej isté predikcie; medzi nimi môžu byť správne aj chybné detekcie. "
        "Vyšší prah filtruje menej isté rámčeky, ale môže vynechať skutočné objekty. "
        "Skóre istoty nie je nameraná presnosť modelu; na tú treba referenčné anotácie.",
        "", "Konkrétna vizuálne overená chyba a jej možné príčiny sú opísané v ../POZOROVANIA.md.",
    ])
    (output_dir / "porovnanie.md").write_text("\n".join(observations) + "\n", encoding="utf-8")
    print(f"\nVýsledky sú uložené v: {output_dir}")
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
