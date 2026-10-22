import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import deque

image = np.array([
    [1, 1, 2, 5, 0, 7, 1, 8, 6],
    [5, 5, 1, 2, 4, 9, 5, 2, 2],
    [5, 20, 36, 98, 5, 4, 7, 8, 4],
    [5, 48, 41, 52, 10, 0, 4, 5, 3],
    [3, 5, 54, 1, 9, 2, 8, 8, 0],
    [2, 6, 56, 6, 3, 3, 7, 5, 5],
    [9, 5, 59, 5, 9, 10, 178, 155, 5],
    [9, 0, 65, 7, 10, 3, 168, 159, 1],
    [5, 1, 54, 9, 5, 4, 1, 9, 3],
    [0, 8, 42, 6, 3, 6, 9, 8, 6],
    [7, 8, 8, 4, 2, 9, 2, 4, 1],
    [10, 9, 4, 5, 1, 10, 5, 6, 2]
], dtype=np.uint8)

threshold = 15
threshold_image = np.where(image > threshold, 255, 0)

k = 3
pixels = image.reshape(-1, 1).astype(float)

kmeans = KMeans(
    n_clusters=k,
    init="random",
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(pixels)
centers = kmeans.cluster_centers_.flatten()
order = np.argsort(centers)

sorted_labels = np.zeros_like(labels)

for new_label, old_label in enumerate(order):
    sorted_labels[labels == old_label] = new_label

kmeans_image = sorted_labels.reshape(image.shape)


def region_growing(img, seed, min_value, max_value):
    rows, cols = img.shape
    mask = np.zeros((rows, cols), dtype=np.uint8)

    sr, sc = seed

    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError("Počiatočný pixel je mimo obrázka.")

    if not (min_value <= img[sr, sc] <= max_value):
        raise ValueError("Počiatočný pixel nespĺňa kritérium.")

    queue = deque([seed])
    mask[sr, sc] = 1

    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    while queue:
        r, c = queue.popleft()

        for dr, dc in neighbors:
            nr = r + dr
            nc = c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                if mask[nr, nc] == 0:
                    if min_value <= img[nr, nc] <= max_value:
                        mask[nr, nc] = 1
                        queue.append((nr, nc))

    return mask


seed_1 = (2, 2)
seed_2 = (6, 6)

region_1 = region_growing(image, seed_1, 15, 120)
region_2 = region_growing(image, seed_2, 120, 255)

region_image = np.zeros_like(image)
region_image[region_1 == 1] = 1
region_image[region_2 == 1] = 2

print("PÔVODNÝ OBRAZ:")
print(image)

print("\nPRAHOVANIE:")
print(threshold_image)

print("\nK-MEANS:")
print(kmeans_image)

print("\nCENTRÁ ZHLUKOV:")
print(np.sort(centers))

print("\nREGION GROWING:")
print(region_image)

print("\nPOČET PIXELOV:")
print("Prahovanie - objekty:", np.sum(threshold_image == 255))
print("Region 1:", np.sum(region_1))
print("Region 2:", np.sum(region_2))

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].imshow(image, cmap="gray_r", vmin=0, vmax=255)
axes[0, 0].set_title("Pôvodný obraz")

axes[0, 1].hist(
    image.flatten(),
    bins=np.arange(-0.5, 256.5, 1),
    color="gray",
    edgecolor="black"
)
axes[0, 1].axvline(
    threshold,
    color="red",
    linestyle="--",
    label=f"Prah = {threshold}"
)
axes[0, 1].set_title("Histogram jasových hodnôt")
axes[0, 1].set_xlabel("Intenzita")
axes[0, 1].set_ylabel("Počet pixelov")
axes[0, 1].legend()

axes[0, 2].imshow(
    threshold_image,
    cmap="gray_r",
    vmin=0,
    vmax=255
)
axes[0, 2].set_title("Prahovanie")

axes[1, 0].imshow(
    kmeans_image,
    cmap="gray_r",
    vmin=0,
    vmax=2
)
axes[1, 0].set_title("K-means (k=3)")

axes[1, 1].imshow(
    region_image,
    cmap="gray_r",
    vmin=0,
    vmax=2
)
axes[1, 1].set_title("Region Growing")

counts = [
    np.sum(threshold_image == 255),
    np.sum(region_1),
    np.sum(region_2)
]

axes[1, 2].bar(
    ["Prahovanie", "Región 1", "Región 2"],
    counts
)
axes[1, 2].set_title("Počet pixelov v objektoch")
axes[1, 2].set_ylabel("Počet pixelov")

for ax in [
    axes[0, 0],
    axes[0, 2],
    axes[1, 0],
    axes[1, 1]
]:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
