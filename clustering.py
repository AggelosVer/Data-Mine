import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

output_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir  = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# ============================================================
# ΒΗΜΑ 1: Φόρτωση & Προεπεξεργασία
# ============================================================
print("=== Βήμα 1: Φόρτωση Δεδομένων ===")
df = pd.read_csv(os.path.join(output_dir, "cleaned_sampled_data.csv"))
print(f"Dataset: {df.shape[0]:,} γραμμές, {df.shape[1]} στήλες")

# Κωδικοποίηση Label για ARI
le = LabelEncoder()
y_true_full = le.fit_transform(df['Label'])
class_names = le.classes_

X_raw = df.drop(columns=['Label']).values

# StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# PCA για οπτικοποίηση (2 διαστάσεις)
print("Εφαρμογή PCA (2 components) για οπτικοποίηση...")
pca_vis = PCA(n_components=2, random_state=42)
X_pca_full = pca_vis.fit_transform(X_scaled)
print(f"  Εξηγούμενη Διακύμανση: {pca_vis.explained_variance_ratio_.sum()*100:.2f}%")

# Χρωματική παλέτα για κλάσεις
palette_true = sns.color_palette("tab20", len(class_names))

def save_cluster_plot(X_pca, labels, title, filename, palette=None, legend_labels=None):
    """Αποθήκευση scatter plot PCA με χρωματισμό clusters."""
    fig, ax = plt.subplots(figsize=(12, 8))
    unique_labels = np.unique(labels)
    if palette is None:
        palette = sns.color_palette("tab20", len(unique_labels))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = legend_labels[i] if legend_labels else f"Cluster {lbl}" if lbl != -1 else "Θόρυβος (DBSCAN)"
        color = 'gray' if lbl == -1 else palette[i % len(palette)]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], s=3, alpha=0.5, label=name)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(f"PCA 1", fontsize=10)
    ax.set_ylabel(f"PCA 2", fontsize=10)
    ax.legend(loc='upper right', markerscale=3, fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=150)
    plt.close()
    print(f"  → Αποθηκεύτηκε: plots/{filename}")

# Plot πραγματικών κλάσεων
save_cluster_plot(X_pca_full, y_true_full, "Πραγματικές Κλάσεις (Ground Truth) - PCA 2D",
                  "clustering_ground_truth.png", palette_true, list(class_names))

# ============================================================
# ΒΗΜΑ 2: K-MEANS (100K δείγματα)
# ============================================================
print("\n=== Βήμα 2: K-Means Clustering ===")

# Αριθμός κλάσεων ως αναφορά
n_true_classes = len(class_names)
print(f"Αριθμός πραγματικών κλάσεων: {n_true_classes}")

# Elbow Method + Silhouette για k ∈ [2, 20]
k_range = range(2, 21)
inertias, silhouettes = [], []

print("Υπολογισμός Elbow + Silhouette για k=2..20 (σε 20K δείγμα)...")
# Χρησιμοποιούμε 20K για γρήγορο elbow search
np.random.seed(42)
idx_elbow = np.random.choice(len(X_scaled), size=20000, replace=False)
X_elbow = X_scaled[idx_elbow]

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=42, max_iter=200)
    km.fit(X_elbow)
    inertias.append(km.inertia_)
    if k <= 16:  # silhouette αργεί για πολλά k
        sil = silhouette_score(X_elbow, km.labels_, sample_size=5000, random_state=42)
        silhouettes.append((k, sil))
    print(f"  k={k:2d} → Inertia: {km.inertia_:.0f}" + (f" | Sil: {sil:.4f}" if k <= 16 else ""))

# Εύρεση elbow μέσω μεθόδου μέγιστης κάθετης απόστασης (Kneedle Algorithm)
k_list = np.array(list(k_range), dtype=float)
inertias_arr = np.array(inertias, dtype=float)

# Κανονικοποίηση αξόνων σε [0, 1] ώστε να μην κυριαρχεί ο ένας άξονας
k_norm = (k_list - k_list[0]) / (k_list[-1] - k_list[0])
inertia_norm = (inertias_arr - inertias_arr[-1]) / (inertias_arr[0] - inertias_arr[-1])

# Κάθετη απόσταση κάθε σημείου από τη γραμμή (πρώτο → τελευταίο σημείο)
line_vec = np.array([k_norm[-1] - k_norm[0], inertia_norm[-1] - inertia_norm[0]])
distances = []
for i in range(len(k_list)):
    point_vec = np.array([k_norm[i] - k_norm[0], inertia_norm[i] - inertia_norm[0]])
    dist = abs(np.cross(line_vec, point_vec)) / np.linalg.norm(line_vec)
    distances.append(dist)

elbow_idx = np.argmax(distances)
elbow_k = int(k_list[elbow_idx])
print(f"\nElbow Point (Kneedle Algorithm): k={elbow_k}")

# Συνδυασμός Elbow + Silhouette:
# Ο elbow ορίζει το ελάχιστο λογικό k (κάτω από αυτό χάνεται πληροφορία).
# Από εκεί και πάνω, επιλέγουμε το k με το μέγιστο Silhouette Score.
sil_dict = dict(silhouettes)
silhouettes_above_elbow = [(k, s) for k, s in silhouettes if k >= elbow_k]
best_k, best_sil = max(silhouettes_above_elbow, key=lambda x: x[1])
print(f"Βέλτιστο k (μέγιστο Silhouette για k≥{elbow_k}): k={best_k} (Sil={best_sil:.4f})")

sil_at_elbow = sil_dict.get(elbow_k, None)

# ---- Elbow + Silhouette Plot (ενημερωμένο) ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Αριστερό: Elbow Method
ax1.plot(list(k_range), inertias, 'bo-', markersize=5)
ax1.scatter([elbow_k], [inertias[elbow_idx]], c='orange', s=80, zorder=5, marker='*',
            label=f'Elbow (k={elbow_k})')
ax1.axvline(x=elbow_k, color='orange', linestyle='--', alpha=0.5)
best_k_inertia_idx = list(k_range).index(best_k)
ax1.scatter([best_k], [inertias[best_k_inertia_idx]], c='red', s=80, zorder=5, marker='D',
            label=f'Βέλτιστο k={best_k}')
ax1.axvline(x=best_k, color='red', linestyle='--', alpha=0.5)
ax1.axvline(x=n_true_classes, color='green', linestyle='--', alpha=0.5,
            label=f'Πραγματικές κλάσεις (k={n_true_classes})')
ax1.set_xlabel('Αριθμός Clusters (k)', fontsize=11)
ax1.set_ylabel('Inertia (SSE)', fontsize=11)
ax1.set_title('Elbow Method για K-Means', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Δεξί: Silhouette Score
sil_k = [s[0] for s in silhouettes]
sil_v = [s[1] for s in silhouettes]
ax2.plot(sil_k, sil_v, 'rs-', markersize=5)
ax2.axvline(x=n_true_classes, color='green', linestyle='--', alpha=0.7,
            label=f'Πραγματικές κλάσεις (k={n_true_classes})')
ax2.axvline(x=elbow_k, color='orange', linestyle='--', alpha=0.5,
            label=f'Elbow (k={elbow_k})')
ax2.scatter([best_k], [best_sil], c='red', s=80, zorder=5, marker='D',
            label=f'Βέλτιστο k={best_k} (Sil={best_sil:.2f})')
ax2.set_xlabel('Αριθμός Clusters (k)', fontsize=11)
ax2.set_ylabel('Silhouette Score', fontsize=11)
ax2.set_title('Silhouette Score ανά k', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle('K-Means: Επιλογή Βέλτιστου k', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "kmeans_elbow_silhouette.png"), dpi=150)
plt.close()
print("  → Αποθηκεύτηκε: plots/kmeans_elbow_silhouette.png")

print(f"\nΒέλτιστο k (Elbow + Silhouette) = {best_k}")
print(f"Εφαρμογή K-Means με k={best_k} στο πλήρες dataset...")

km_final = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42, max_iter=300)
km_labels = km_final.fit_predict(X_scaled)

sil_km  = silhouette_score(X_scaled, km_labels, sample_size=10000, random_state=42)
dbi_km  = davies_bouldin_score(X_scaled[:10000], km_labels[:10000])
ari_km  = adjusted_rand_score(y_true_full, km_labels)
print(f"  Silhouette Score: {sil_km:.4f}")
print(f"  Davies-Bouldin Index: {dbi_km:.4f}")
print(f"  Adjusted Rand Index (vs Label): {ari_km:.4f}")

save_cluster_plot(X_pca_full, km_labels,
                  f"K-Means (k={best_k}) – PCA 2D",
                  "kmeans_pca.png")
# ΒΗΜΑ 3: ΙΕΡΑΡΧΙΚΗ ΟΜΑΔΟΠΟΙΗΣΗ (5K δείγματα)
# ============================================================
print("\n=== Βήμα 3: Ιεραρχική Ομαδοποίηση (Hierarchical Clustering) ===")
HIER_N = 5000
print(f"Χρήση {HIER_N:,} τυχαίων δειγμάτων (O(N²) μνήμη)")

np.random.seed(42)
idx_hier = np.random.choice(len(X_scaled), size=HIER_N, replace=False)
X_hier  = X_scaled[idx_hier]
y_hier  = y_true_full[idx_hier]
X_pca_hier = X_pca_full[idx_hier]

linkage_methods = ['ward', 'complete', 'average']
hier_results = {}

for method in linkage_methods:
    print(f"\nΕκτέλεση Ιεραρχικής με μέθοδο σύνδεσης: {method}...")
    
    # Παραγωγή Dendrogram
    Z = linkage(X_hier, method=method)
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=50,
               leaf_rotation=90, leaf_font_size=8, show_contracted=True)
    ax.set_title(f"Dendrogram Ιεραρχικής Ομαδοποίησης ({method}, τελευταία 50 merges)", fontsize=12)
    ax.set_xlabel("Δείγματα / Clusters", fontsize=10)
    ax.set_ylabel(f"Απόσταση ({method})", fontsize=10)
    plt.tight_layout()
    dendrogram_filename = f"hierarchical_dendrogram_{method}.png"
    plt.savefig(os.path.join(plots_dir, dendrogram_filename), dpi=150)
    plt.close()
    print(f"  → Αποθηκεύτηκε: plots/{dendrogram_filename}")

    # Εκπαίδευση μοντέλου με k = best_k (ίδιο με K-Means για σύγκριση)
    agg = AgglomerativeClustering(n_clusters=best_k, linkage=method)
    hier_labels = agg.fit_predict(X_hier)

    # Υπολογισμός μετρικών
    try:
        sil_hier = silhouette_score(X_hier, hier_labels, sample_size=3000, random_state=42)
    except ValueError:
        sil_hier = silhouette_score(X_hier, hier_labels)
    dbi_hier = davies_bouldin_score(X_hier, hier_labels)
    ari_hier = adjusted_rand_score(y_hier, hier_labels)
    
    print(f"  [{method}] Silhouette Score: {sil_hier:.4f}")
    print(f"  [{method}] Davies-Bouldin Index: {dbi_hier:.4f}")
    print(f"  [{method}] Adjusted Rand Index (vs Label): {ari_hier:.4f}")
    
    hier_results[method] = {
        'labels': hier_labels,
        'sil': sil_hier,
        'dbi': dbi_hier,
        'ari': ari_hier
    }
    
    # Αποθήκευση PCA plot
    pca_filename = f"hierarchical_pca_{method}.png"
    save_cluster_plot(X_pca_hier, hier_labels,
                      f"Ιεραρχική Ομαδοποίηση ({method}, k={best_k}, N={HIER_N:,}) – PCA 2D",
                      pca_filename)

# Επιλογή καλύτερης μεθόδου ιεραρχικής βάσει Silhouette Score
best_hier_method = max(hier_results, key=lambda m: hier_results[m]['sil'])
print(f"\nΒέλτιστη μέθοδος σύνδεσης: {best_hier_method} (Silhouette: {hier_results[best_hier_method]['sil']:.4f})")

sil_hier_best = hier_results[best_hier_method]['sil']
dbi_hier_best = hier_results[best_hier_method]['dbi']
ari_hier_best = hier_results[best_hier_method]['ari']
hier_labels_best = hier_results[best_hier_method]['labels']

# ============================================================
# ΒΗΜΑ 4: DBSCAN GRID SEARCH (20K δείγματα)
# ============================================================
print("\n=== Βήμα 4: DBSCAN Grid Search ===")
DBSCAN_N = 20000
print(f"Χρήση {DBSCAN_N:,} τυχαίων δειγμάτων")

np.random.seed(42)
idx_db = np.random.choice(len(X_scaled), size=DBSCAN_N, replace=False)
X_db   = X_scaled[idx_db]
y_db   = y_true_full[idx_db]
X_pca_db = X_pca_full[idx_db]

# Ορισμός πλέγματος παραμέτρων
eps_grid = [1.0, 1.5, 2.0, 2.5]
min_samples_grid = [5, 10, 15, 20]
dbscan_runs = []

print("Έναρξη Grid Search για DBSCAN...")
for eps in eps_grid:
    for min_samples in min_samples_grid:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_db)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        noise_pct = (n_noise / len(labels)) * 100
        
        sil, dbi, ari = float('nan'), float('nan'), float('nan')
        if n_clusters >= 2:
            mask_valid = labels != -1
            if mask_valid.sum() > 100:
                try:
                    sil = silhouette_score(X_db[mask_valid], labels[mask_valid],
                                           sample_size=min(3000, mask_valid.sum()), random_state=42)
                except ValueError:
                    try:
                        sil = silhouette_score(X_db[mask_valid], labels[mask_valid])
                    except ValueError:
                        sil = float('nan')
                dbi = davies_bouldin_score(X_db[mask_valid], labels[mask_valid])
                ari = adjusted_rand_score(y_db[mask_valid], labels[mask_valid])
        
        print(f"  eps={eps:.1f}, min_samples={min_samples:2d} → Clusters: {n_clusters:3d} | Noise: {noise_pct:5.1f}%" + 
              (f" | Sil (no noise): {sil:.4f}" if not np.isnan(sil) else ""))
        
        dbscan_runs.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'noise_pct': noise_pct,
            'sil': sil,
            'dbi': dbi,
            'ari': ari,
            'labels': labels
        })

# Επιλογή βέλτιστης παραμέτρου DBSCAN
# Κριτήριο: clusters >= 2, noise_pct < 50%, και μέγιστο Silhouette score
valid_runs = [r for r in dbscan_runs if r['n_clusters'] >= 2 and r['noise_pct'] < 50.0 and not np.isnan(r['sil'])]
if not valid_runs:
    # Αν κανένα δεν ικανοποιεί τα κριτήρια, παίρνουμε οποιοδήποτε έχει clusters >= 2
    valid_runs = [r for r in dbscan_runs if r['n_clusters'] >= 2 and not np.isnan(r['sil'])]

if valid_runs:
    best_dbscan_run = max(valid_runs, key=lambda x: x['sil'])
else:
    # Fallback στην πρώτη εκτέλεση
    best_dbscan_run = dbscan_runs[0]

eps_db_best = best_dbscan_run['eps']
min_samples_db_best = best_dbscan_run['min_samples']
n_clusters_db_best = best_dbscan_run['n_clusters']
noise_pct_db_best = best_dbscan_run['noise_pct']
sil_db_best = best_dbscan_run['sil']
dbi_db_best = best_dbscan_run['dbi']
ari_db_best = best_dbscan_run['ari']
db_labels_best = best_dbscan_run['labels']

print(f"\nΒέλτιστο DBSCAN: eps={eps_db_best:.1f}, min_samples={min_samples_db_best} "
      f"(Clusters: {n_clusters_db_best}, Noise: {noise_pct_db_best:.1f}%, Silhouette: {sil_db_best:.4f})")

# Αποθήκευση PCA plot για το βέλτιστο DBSCAN
save_cluster_plot(X_pca_db, db_labels_best,
                  f"DBSCAN Βέλτιστο (eps={eps_db_best}, min_samples={min_samples_db_best}, N={DBSCAN_N:,}) – PCA 2D",
                  "dbscan_pca.png")

# ============================================================
# ΒΗΜΑ 5: Συγκριτικό Γράφημα Μετρικών
# ============================================================
print("\n=== Βήμα 5: Συγκριτικό Γράφημα ===")

algo_names = ['K-Means', f'Hierarchical ({best_hier_method})', 'DBSCAN']
sil_vals   = [sil_km, sil_hier_best, sil_db_best if not np.isnan(sil_db_best) else 0]
dbi_vals   = [dbi_km, dbi_hier_best, dbi_db_best if not np.isnan(dbi_db_best) else 0]
ari_vals   = [ari_km, ari_hier_best, ari_db_best if not np.isnan(ari_db_best) else 0]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['#4C72B0', '#DD8452', '#55A868']

for ax, vals, metric in zip(axes,
                             [sil_vals, dbi_vals, ari_vals],
                             ['Silhouette Score\n(↑ καλύτερο)',
                              'Davies-Bouldin Index\n(↓ καλύτερο)',
                              'Adjusted Rand Index\n(↑ καλύτερο)']):
    bars = ax.bar(algo_names, vals, color=colors, alpha=0.85, edgecolor='white')
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.set_ylabel('Τιμή', fontsize=10)
    for bar in bars:
        val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.01 if val >= 0 else -0.04),
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(min(0, min(vals) * 1.2), max(vals) * 1.2 + 0.05)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Σύγκριση Αλγορίθμων Ομαδοποίησης (Βέλτιστες Παράμετροι)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "clustering_comparison.png"), dpi=150)
plt.close()
print("  → Αποθηκεύτηκε: plots/clustering_comparison.png")
