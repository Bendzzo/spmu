
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from time import perf_counter


# 1. Extrakcia priznakov pomocou HOG
def extract_hog_features(images):
    features = []

    for image in images:
        # Odstranenie dimenzie kanala (28, 28, 1) -> (28, 28)
        image = image.squeeze()

        # Extrakcia hran a ich orientacii
        feature_vector = hog(
            image,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        features.append(feature_vector)

    return np.array(features)


# 2. Trenovanie a testovanie SVM
def train_hog_svm(x_train, y_train, x_test, y_test):
    print("\n=== HOG + SVM ===")

    # Extrakcia priznakov
    print("Extrahujem HOG priznaky...")
    start = perf_counter()

    train_features = extract_hog_features(x_train)
    test_features = extract_hog_features(x_test)

    print("Tvar priznakov:", train_features.shape)
    print(f"Extrakcia trvala: {perf_counter() - start:.2f} s")

    # Vytvorenie SVM klasifikatora
    model = LinearSVC(
        C=1.0,
        dual=False,
        max_iter=2000,
        random_state=42
    )

    # Trenovanie
    print("Trenujem SVM model...")
    start = perf_counter()

    model.fit(train_features, y_train)

    print(f"Trenovanie trvalo: {perf_counter() - start:.2f} s")

    # Predikcia testovacich obrazkov
    predictions = model.predict(test_features)

    # Vyhodnotenie presnosti
    accuracy = accuracy_score(y_test, predictions)

    print(f"Presnost HOG + SVM: {accuracy * 100:.2f} %")

    return accuracy, predictions
