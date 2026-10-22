
import numpy as np
import matplotlib.pyplot as plt

from dataset import load_mnist
from cnn_model import create_cnn
from hog_svm_model import train_hog_svm


# 1. Nacitanie datasetu MNIST
x_train, y_train, x_test, y_test = load_mnist()

print("Trenovacie data:", x_train.shape)
print("Testovacie data:", x_test.shape)


# 2. Nastavenie experimentov
repetitions = 5
accuracies = []
histories = []


# 3. Opakovane trenovanie CNN
for i in range(repetitions):
    print(f"\n--- Experiment {i + 1}/{repetitions} ---")

    # Vytvorenie noveho modelu s novymi vahami
    model = create_cnn()

    # Trenovanie modelu
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    # Vyhodnotenie na testovacich datach
    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    # Ulozenie vysledkov
    accuracies.append(accuracy)
    histories.append(history.history)

    print(f"Presnost experimentu {i + 1}: {accuracy * 100:.2f} %")


# 4. Statisticke vyhodnotenie
average_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print("\n========== VYSLEDKY CNN ==========")

for i, accuracy in enumerate(accuracies):
    print(f"Experiment {i + 1}: {accuracy * 100:.2f} %")

print(f"\nPriemerna presnost: {average_accuracy * 100:.2f} %")
print(f"Smerodajna odchylka: {std_accuracy * 100:.3f} %")


# 5. Graf presnosti jednotlivych experimentov
plt.figure(figsize=(9, 5))

plt.bar(
    range(1, repetitions + 1),
    np.array(accuracies) * 100
)

plt.axhline(
    average_accuracy * 100,
    color="red",
    linestyle="--",
    label=f"Priemer: {average_accuracy * 100:.2f} %"
)

plt.xlabel("Cislo experimentu")
plt.ylabel("Presnost (%)")
plt.title("Presnost CNN pri opakovanom trenovani")
plt.xticks(range(1, repetitions + 1))
plt.ylim(0, 100)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# 6. Graf priebehu trenovania posledneho modelu
last_history = histories[-1]

plt.figure(figsize=(9, 5))

plt.plot(
    last_history["accuracy"],
    label="Trenovacia presnost"
)

plt.plot(
    last_history["val_accuracy"],
    label="Validacna presnost"
)

plt.xlabel("Epocha")
plt.ylabel("Presnost")
plt.title("Priebeh trenovania posledneho CNN modelu")
plt.xticks(range(5), range(1, 6))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 7. Ukazka predikcii posledneho modelu
predictions = model.predict(x_test[:10], verbose=0)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    predicted_digit = np.argmax(predictions[i])
    actual_digit = y_test[i]

    ax.imshow(x_test[i].squeeze(), cmap="gray")

    ax.set_title(
        f"Predikcia: {predicted_digit}\n"
        f"Skutocnost: {actual_digit}"
    )

    ax.axis("off")

plt.suptitle("Ukazka klasifikacie MNIST")
plt.tight_layout()
plt.show()



# 8. Trenovanie druheho modelu - HOG + SVM
svm_accuracy, svm_predictions = train_hog_svm(
    x_train,
    y_train,
    x_test,
    y_test
)


# 9. Porovnanie CNN a SVM
print("\n========== POROVNANIE MODELOV ==========")

print(f"CNN priemerna presnost: {average_accuracy * 100:.2f} %")
print(f"HOG + SVM presnost:    {svm_accuracy * 100:.2f} %")

# Graf porovnania
model_names = ["CNN", "HOG + SVM"]
model_accuracies = [
    average_accuracy * 100,
    svm_accuracy * 100
]

plt.figure(figsize=(8, 5))
plt.bar(model_names, model_accuracies)

plt.title("Porovnanie klasifikacnych modelov")
plt.ylabel("Presnost (%)")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)

for i, value in enumerate(model_accuracies):
    plt.text(
        i, value,
        f"{value:.2f} %",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()
