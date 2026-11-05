"""Cvičenie 6: načítanie, kontrola, uloženie a porovnanie EEG meraní.

Spustenie: python main.py (alebo python main.py --no-show bez okna grafu).
"""

import argparse
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
GROUPS = ("Z", "O", "N", "F", "S")
DESCRIPTIONS = {
    "Z": "zdraví, otvorené oči",
    "O": "zdraví, zatvorené oči",
    "N": "mimo záchvatu, opačná hemisféra",
    "F": "mimo záchvatu, epileptogénna oblasť",
    "S": "počas epileptického záchvatu",
}
SAMPLING_RATE_HZ = 173.61


def load_measurement(path):
    """Overí jeden textový súbor: práve jedna konečná číselná hodnota na riadok."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError(f"Prázdny súbor: {path}")
    for line_number, line in enumerate(lines, start=1):
        if len(line.split()) != 1:
            raise ValueError(
                f"{path}, riadok {line_number}: očakáva sa jedna hodnota; "
                "prázdny riadok alebo viac stĺpcov nie je povolených."
            )
    try:
        values = np.loadtxt(lines, dtype=np.float64, ndmin=1)
    except ValueError as error:
        raise ValueError(f"Nečíselná alebo chýbajúca hodnota v {path}: {error}") from error
    if not np.isfinite(values).all():
        raise ValueError(f"Súbor obsahuje NaN alebo nekonečno: {path}")
    return values


def load_dataset(data_dir):
    """Vytvorí maticu meraní a rovnako zoradené polia s pôvodom riadkov."""
    records, groups, filenames = [], [], []
    for group in GROUPS:
        folder = data_dir / group
        if not folder.is_dir():
            raise ValueError(f"Chýba priečinok skupiny: {folder}")
        files = sorted(
            (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt"),
            key=lambda p: p.name.lower(),
        )
        if not files:
            raise ValueError(f"Priečinok neobsahuje TXT merania: {folder}")
        for path in files:
            records.append(load_measurement(path))
            groups.append(group)
            filenames.append(path.name)

    lengths = np.array([record.size for record in records])
    unique_lengths, counts = np.unique(lengths, return_counts=True)
    if unique_lengths.size != 1:
        details = dict(zip(unique_lengths.tolist(), counts.tolist()))
        raise ValueError(
            f"Merania majú rôzne dĺžky (počet vzoriek: počet meraní): {details}. "
            "Pred vytvorením matice zvoľte spoločný časový úsek, napríklad "
            "orezanie na najkratší záznam. Automatické orezanie sa nevykonalo, "
            "pretože by zahodilo časť meraní."
        )

    return {
        "signals": np.stack(records),
        "groups": np.array(groups),
        "filenames": np.array(filenames),
        "is_seizure": np.array(groups) == "S",
        "sampling_rate_hz": np.array(SAMPLING_RATE_HZ),
    }


def save_and_verify(dataset, output_path):
    """Uloží aj metadáta a porovná každé pole po opätovnom načítaní."""
    np.savez_compressed(output_path, **dataset)
    # Textové polia majú Unicode dtype, preto nepotrebujeme pickle.
    with np.load(output_path, allow_pickle=False) as restored:
        if set(restored.files) != set(dataset):
            raise ValueError("Uložený archív obsahuje nesprávne polia.")
        for key, original in dataset.items():
            loaded = restored[key]
            if loaded.dtype != original.dtype or not np.array_equal(loaded, original):
                raise ValueError(f"Overenie uloženého poľa {key!r} zlyhalo.")


def create_summary(dataset):
    """Textový prehľad kontrol a amplitúd, vypočítaný zo všetkých meraní."""
    signals = dataset["signals"]
    groups = dataset["groups"]
    lines = [
        "KONTROLA EEG DÁT",
        f"Rozmery matice: {signals.shape} (merania × vzorky)",
        f"Dátový typ matice: {signals.dtype}",
        f"Všetky hodnoty sú celočíselné: {bool(np.all(signals == np.trunc(signals)))}",
        f"Počet chýbajúcich/nekonečných hodnôt: {int(np.count_nonzero(~np.isfinite(signals)))}",
        f"Počet vzoriek každého merania: {signals.shape[1]}",
        f"Vzorkovacia frekvencia: {SAMPLING_RATE_HZ} Hz",
        f"Dĺžka záznamu N/fs: {signals.shape[1] / SAMPLING_RATE_HZ:.3f} s",
        f"Čas poslednej vzorky (N-1)/fs: {(signals.shape[1] - 1) / SAMPLING_RATE_HZ:.3f} s",
        "Poradie skupín: " + ", ".join(GROUPS),
        "Indexy riadkov sú číslované od nuly.",
        "Amplitúdy sú pôvodné číselné hodnoty súborov, bez normalizácie a filtrácie.",
        "",
        "PREHĽAD SKUPÍN",
        "Skupina | počet | riadky | minimum | maximum | medián rozpätia max-min",
    ]
    for group in GROUPS:
        indices = np.flatnonzero(groups == group)
        subset = signals[indices]
        lines.append(
            f"{group} | {indices.size} | {indices[0]}-{indices[-1]} | "
            f"{subset.min():.0f} | {subset.max():.0f} | "
            f"{np.median(np.ptp(subset, axis=1)):.1f}"
        )
        lines.append("  " + DESCRIPTIONS[group])
    lines.extend(["", "MERANIA V GRAFOCH (prvý súbor každej skupiny)",
                  "Súbor | minimum | maximum | smerodajná odchýlka"])
    for group in GROUPS:
        index = np.flatnonzero(groups == group)[0]
        record = signals[index]
        lines.append(
            f"{dataset['filenames'][index]} | {record.min():.0f} | "
            f"{record.max():.0f} | {record.std():.2f}"
        )
    lines.extend(["", "NPZ archív bol opäť načítaný; všetky polia sa presne zhodujú."])
    return "\n".join(lines) + "\n"


def plot_measurements(dataset, output_path, show=True):
    """Päť priebehov a ich detaily s rovnakou mierkou amplitúdy."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signals = dataset["signals"]
    time = np.arange(signals.shape[1]) / float(dataset["sampling_rate_hz"])
    selected = [np.flatnonzero(dataset["groups"] == group)[0] for group in GROUPS]
    limit = max(1.0, float(np.abs(signals[selected]).max())) * 1.05
    fig, axes = plt.subplots(5, 2, figsize=(15, 12), sharey=True, layout="constrained")
    for row, (group, index) in enumerate(zip(GROUPS, selected)):
        color = "#bd3038" if group == "S" else "#216a99"
        for col, ax in enumerate(axes[row]):
            ax.plot(time, signals[index], linewidth=0.65, color=color)
            ax.set_ylim(-limit, limit)
            ax.set_xlim(0, time[-1] if col == 0 else min(5.0, time[-1]))
            ax.set_title(
                f"{dataset['filenames'][index]} - {DESCRIPTIONS[group]}"
                if col == 0 else f"{group}: detail prvých 5 sekúnd",
                fontsize=10,
            )
            ax.set_xlabel("Čas [s]")
            ax.grid(alpha=0.25)
        axes[row, 0].set_ylabel("Amplitúda\n[hodnoty zo súboru]")
    fig.suptitle("EEG: rovnaká mierka amplitúdy vo všetkých grafoch", fontsize=15)
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "EEG Records")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "vystupy")
    parser.add_argument("--no-show", action="store_true", help="Uložiť graf bez otvorenia okna.")
    args = parser.parse_args()

    try:
        dataset = load_dataset(args.data_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        archive_path = args.output_dir / "eeg_merania.npz"
        save_and_verify(dataset, archive_path)
        summary = create_summary(dataset)
        (args.output_dir / "kontrola_dat.txt").write_text(summary, encoding="utf-8")
        plot_measurements(dataset, args.output_dir / "porovnanie_eeg.png", show=not args.no_show)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Chyba: {error}\n")

    print(summary)
    print(f"Archív, graf a prehľad sú uložené v: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
