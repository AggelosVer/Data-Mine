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

output_dir = r"C:\Users\user\Documents\GitHub\Data Mine"
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

# Elbow Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(list(k_range), inertias, 'bo-', markersize=5)
ax1.set_xlabel('Αριθμός Clusters (k)', fontsize=11)
ax1.set_ylabel('Inertia (SSE)', fontsize=11)
ax1.set_title('Elbow Method για K-Means', fontsize=12)
ax1.grid(True, alpha=0.3)

sil_k = [s[0] for s in silhouettes]
sil_v = [s[1] for s in silhouettes]
ax2.plot(sil_k, sil_v, 'rs-', markersize=5)
ax2.axvline(x=n_true_classes, color='green', linestyle='--', alpha=0.7,
            label=f'Πραγματικές κλάσεις (k={n_true_classes})')
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

# Βέλτιστο k = αυτό με το μέγιστο silhouette
best_k = max(silhouettes, key=lambda x: x[1])[0]
print(f"\nΒέλτιστο k (μέγιστο Silhouette) = {best_k}")
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
    sil_hier = silhouette_score(X_hier, hier_labels, sample_size=3000, random_state=42)
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
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(X_db)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        noise_pct = (n_noise / len(labels)) * 100
        
        sil, dbi, ari = float('nan'), float('nan'), float('nan')
        if n_clusters >= 2:
            mask_valid = labels != -1
            if mask_valid.sum() > 100:
                sil = silhouette_score(X_db[mask_valid], labels[mask_valid],
                                       sample_size=min(3000, mask_valid.sum()), random_state=42)
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

# ============================================================
# ΒΗΜΑ 6: Αναφορά Report_Q3.md
# ============================================================
print("\n=== Βήμα 6: Παραγωγή Report_Q3.md ===")

with open(os.path.join(output_dir, "Report_Q3.md"), "w", encoding="utf-8") as rf:
    rf.write("# Αναφορά Ερωτήματος 3 – Ομαδοποίηση (Clustering)\n\n")
    rf.write("Αυτή η αναφορά παρουσιάζει τα αποτελέσματα εφαρμογής αλγορίθμων μη-επιβλεπόμενης μάθησης "
             "(unsupervised learning) στο σύνολο δεδομένων **CIC-IDS-2017**, "
             "με στόχο την ανακάλυψη φυσικών ομαδοποιήσεων στη δικτυακή κίνηση.\n\n")

    rf.write("## 1. Προεπεξεργασία Δεδομένων\n\n")
    rf.write("### 1.1 Σύνολο Δεδομένων\n")
    rf.write("Χρησιμοποιήθηκε το καθαρισμένο δείγμα `cleaned_sampled_data.csv` (100.000 εγγραφές, 48 χαρακτηριστικά). "
             "Σε αντίθεση με το Ερώτημα 2, η ομαδοποίηση πραγματοποιήθηκε **χωρίς τη χρήση της μεταβλητής Label** "
             "— η Label χρησιμοποιείται μόνο για εκ των υστέρων αξιολόγηση (ground truth comparison).\n\n")
    rf.write("### 1.2 Κανονικοποίηση\n")
    rf.write("Εφαρμόστηκε `StandardScaler` στο σύνολο των χαρακτηριστικών, καθώς οι αλγόριθμοι ομαδοποίησης "
             "(K-Means, DBSCAN) είναι εξαιρετικά ευαίσθητοι στην κλίμακα των δεδομένων.\n\n")
    rf.write("### 1.3 Μείωση Διαστασιμότητας (PCA)\n")
    rf.write(f"Για την οπτικοποίηση εφαρμόστηκε PCA με 2 components, που εξηγεί το "
             f"**{pca_vis.explained_variance_ratio_.sum()*100:.2f}%** της συνολικής διακύμανσης. "
             f"Η PCA χρησιμοποιείται **μόνο για οπτικοποίηση** — η ομαδοποίηση γίνεται στον πλήρη χώρο 48 διαστάσεων.\n\n")
    rf.write("### 1.4 Μέγεθος Dataset ανά Αλγόριθμο\n")
    rf.write("| Αλγόριθμος | Δείγματα | Λόγος Περιορισμού |\n|:---|:---:|:---|\n")
    rf.write(f"| K-Means | 100.000 | Αποδοτικός, κλιμακώνεται καλά |\n")
    rf.write(f"| Hierarchical (Ward/Complete/Average) | {HIER_N:,} | Απαιτεί O(N²) μνήμη — αδύνατο σε 100K |\n")
    rf.write(f"| DBSCAN | {DBSCAN_N:,} | Ισορροπία ταχύτητας/αντιπροσωπευτικότητας |\n\n")

    rf.write("## 2. K-Means Clustering\n\n")
    rf.write("### 2.1 Επιλογή Βέλτιστου k\n")
    rf.write("Χρησιμοποιήθηκαν δύο συμπληρωματικές μέθοδοι για την επιλογή του βέλτιστου αριθμού clusters:\n\n")
    rf.write("* **Elbow Method:** Αναζήτηση της γωνίας (elbow) στη γραφική απεικόνιση του SSE (inertia) "
             "συναρτήσει του k. Η απότομη μείωση σταματά, υποδεικνύοντας το βέλτιστο k.\n")
    rf.write("* **Silhouette Score:** Μέτρο συνεκτικότητας και διαχωρισμού των clusters. "
             "Τιμές κοντά στο 1 = καλά ξεχωριστά clusters. Επιλέχθηκε το k με τη μέγιστη τιμή.\n\n")
    rf.write(f"Τα γραφήματα αποθηκεύτηκαν στο `plots/kmeans_elbow_silhouette.png`.\n")
    rf.write(f"**Βέλτιστο k = {best_k}** (μέγιστο Silhouette Score).\n\n")
    rf.write("### 2.2 Υλοποίηση\n")
    rf.write(f"* **Αλγόριθμος αρχικοποίησης:** `k-means++` (αποφυγή κακής τυχαίας αρχικοποίησης)\n")
    rf.write(f"* **n_init=10:** Εκτέλεση 10 φορών με διαφορετικά αρχικά centroids, επιλογή καλύτερου\n")
    rf.write(f"* **max_iter=300:** Μέγιστος αριθμός επαναλήψεων σύγκλισης\n\n")
    rf.write("### 2.3 Αποτελέσματα\n")
    rf.write(f"| Μετρική | Τιμή |\n|:---|:---:|\n")
    rf.write(f"| Silhouette Score | {sil_km:.4f} |\n")
    rf.write(f"| Davies-Bouldin Index | {dbi_km:.4f} |\n")
    rf.write(f"| Adjusted Rand Index (vs Label) | {ari_km:.4f} |\n\n")
    rf.write("Η οπτικοποίηση PCA αποθηκεύτηκε στο `plots/kmeans_pca.png`.\n\n")

    rf.write("## 3. Ιεραρχική Ομαδοποίηση (Hierarchical Clustering) – Σύγκριση Μεθόδων Σύνδεσης\n\n")
    rf.write("### 3.1 Μέθοδος και Διερεύνηση Linkages\n")
    rf.write(f"Χρησιμοποιήθηκε **Agglomerative Clustering** (bottom-up) για $k={best_k}$. "
             f"Λόγω της τετραγωνικής πολυπλοκότητας $O(N^2)$, η ανάλυση περιορίστηκε σε {HIER_N:,} δείγματα. "
             f"Πραγματοποιήθηκε πειραματισμός με τρεις διαφορετικές μεθόδους σύνδεσης (linkage methods):\n\n")
    rf.write("* **Ward:** Ελαχιστοποιεί τη συνολική διακύμανση εντός των clusters. Τείνει να δημιουργεί clusters παρόμοιου μεγέθους.\n")
    rf.write("* **Complete Linkage:** Συνδέει clusters με βάση τη μέγιστη απόσταση μεταξύ των σημείων τους.\n")
    rf.write("* **Average Linkage:** Συνδέει clusters με βάση τη μέση απόσταση μεταξύ όλων των ζευγών σημείων τους.\n\n")
    
    rf.write("### 3.2 Αποτελέσματα Σύγκρισης Linkage Methods\n\n")
    rf.write("| Μέθοδος Σύνδεσης | Silhouette Score ↑ | Davies-Bouldin Index ↓ | Adjusted Rand Index (vs Label) ↑ |\n")
    rf.write("|:---|:---:|:---:|:---:|\n")
    for method in linkage_methods:
        res = hier_results[method]
        rf.write(f"| **{method}** | {res['sil']:.4f} | {res['dbi']:.4f} | {res['ari']:.4f} |\n")
    rf.write("\n")
    
    rf.write("### 3.3 Dendrograms & PCA Plots\n")
    rf.write("* Τα δενδρογράμματα αποθηκεύτηκαν στα αρχεία:\n")
    for method in linkage_methods:
        rf.write(f"  * `plots/hierarchical_dendrogram_{method}.png`\n")
    rf.write("* Οι αντίστοιχες οπτικοποιήσεις PCA αποθηκεύτηκαν στα:\n")
    for method in linkage_methods:
        rf.write(f"  * `plots/hierarchical_pca_{method}.png`\n")
    rf.write(f"\n**Παρατήρηση:** Η μέθοδος **{best_hier_method}** παρουσίασε το υψηλότερο Silhouette Score ({sil_hier_best:.4f}) "
             f"και χρησιμοποιείται ως η βέλτιστη εκπροσώπηση της Ιεραρχικής ομαδοποίησης.\n\n")

    rf.write("## 4. DBSCAN – Διερεύνηση Υπερπαραμέτρων (Grid Search)\n\n")
    rf.write("### 4.1 Μεθοδολογία Grid Search\n")
    rf.write("Για τον αλγόριθμο DBSCAN πραγματοποιήθηκε συστηματικός πειραματισμός σε δείγμα 20.000 παρατηρήσεων "
             "με τις ακόλουθες υπερπαράμετρους:\n")
    rf.write("* **Epsilon (eps):** Ακτίνα γειτνίασης $\in \{1.0, 1.5, 2.0, 2.5\}$\n")
    rf.write("* **Min Samples (min_samples):** Ελάχιστος αριθμός σημείων $\in \{5, 10, 15, 20\}$\n\n")
    rf.write("Καταγράφεται πώς οι παράμετροι επηρεάζουν τον αριθμό των συστάδων που προκύπτουν, το ποσοστό θορύβου, "
             "καθώς και τις μετρικές ποιότητας των συστάδων (στα σημεία εκτός θορύβου).\n\n")
    
    rf.write("### 4.2 Αποτελέσματα Grid Search\n\n")
    rf.write("| Epsilon (eps) | Min Samples | Clusters | Θόρυβος (%) | Silhouette ↑ | Davies-Bouldin ↓ | ARI (vs Label) ↑ |\n")
    rf.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    for run in dbscan_runs:
        sil_str = f"{run['sil']:.4f}" if not np.isnan(run['sil']) else "N/A"
        dbi_str = f"{run['dbi']:.4f}" if not np.isnan(run['dbi']) else "N/A"
        ari_str = f"{run['ari']:.4f}" if not np.isnan(run['ari']) else "N/A"
        rf.write(f"| {run['eps']:.1f} | {run['min_samples']} | {run['n_clusters']} | {run['noise_pct']:.1f}% | {sil_str} | {dbi_str} | {ari_str} |\n")
    rf.write("\n")
    
    rf.write("### 4.3 Ανάλυση Επίδρασης των Παραμέτρων\n")
    rf.write("* **Επίδραση του Epsilon (eps):** Καθώς το `eps` αυξάνεται, η ακτίνα γειτνίασης μεγαλώνει, "
             "με αποτέλεσμα περισσότερα σημεία να θεωρούνται core ή border points. Αυτό οδηγεί σε **δραστική μείωση του θορύβου** "
             "και συγχώνευση μικρών συστάδων σε μεγαλύτερες. Αντίθετα, μικρές τιμές `eps` (π.χ. 1.0) αφήνουν μεγάλο ποσοστό "
             "σημείων ως θόρυβο.\n")
    rf.write("* **Επίδραση του Min Samples:** Αυξάνοντας το `min_samples`, οι απαιτήσεις πυκνότητας γίνονται πιο αυστηρές. "
             "Αυτό οδηγεί σε **αύξηση του θορύβου** και λιγότερα clusters, καθώς πολλά οριακά σημεία απορρίπτονται.\n\n")
             
    rf.write(f"### 4.4 Επιλογή Βέλτιστου Μοντέλου\n")
    rf.write(f"Ως βέλτιστος συνδυασμός επιλέχθηκε ο: **eps={eps_db_best:.1f}, min_samples={min_samples_db_best}** "
             f"ο οποίος εντόπισε **{n_clusters_db_best}** clusters με ποσοστό θορύβου **{noise_pct_db_best:.1f}%**.\n")
    rf.write(f"* Silhouette Score (χωρίς θόρυβο): **{sil_db_best:.4f}**\n")
    rf.write(f"* Davies-Bouldin Index (χωρίς θόρυβο): **{dbi_db_best:.4f}**\n")
    rf.write(f"* Adjusted Rand Index (vs Label, χωρίς θόρυβο): **{ari_db_best:.4f}**\n\n")
    rf.write("Η οπτικοποίηση PCA αποθηκεύτηκε στο `plots/dbscan_pca.png`.\n\n")

    rf.write("## 5. Συγκριτική Ανάλυση και Συμπεράσματα (Βέλτιστα Μοντέλα)\n\n")
    rf.write("### 5.1 Σύγκριση Μετρικών\n\n")
    rf.write("| Αλγόριθμος | Silhouette ↑ | Davies-Bouldin ↓ | ARI ↑ | Clusters |\n")
    rf.write("|:---|:---:|:---:|:---:|:---:|\n")
    rf.write(f"| **K-Means (k={best_k})** | {sil_km:.4f} | {dbi_km:.4f} | {ari_km:.4f} | {best_k} |\n")
    rf.write(f"| **Hierarchical ({best_hier_method}, k={best_k})** | {sil_hier_best:.4f} | {dbi_hier_best:.4f} | {ari_hier_best:.4f} | {best_k} |\n")
    
    sil_str = f"{sil_db_best:.4f}" if not np.isnan(sil_db_best) else "N/A"
    dbi_str = f"{dbi_db_best:.4f}" if not np.isnan(dbi_db_best) else "N/A"
    ari_str = f"{ari_db_best:.4f}" if not np.isnan(ari_db_best) else "N/A"
    rf.write(f"| **DBSCAN (eps={eps_db_best:.1f}, min_pts={min_samples_db_best})** | {sil_str} | {dbi_str} | {ari_str} | {n_clusters_db_best} |\n\n")

    rf.write("### 5.2 Ερμηνεία Μετρικών σε σχέση με τη μεταβλητή Label\n")
    rf.write("* **K-Means και Ιεραρχική:** Και οι δύο αλγόριθμοι συγκλίνουν στο $k=2$ ως τη βέλτιστη φυσική ομαδοποίηση. "
             "Μελετώντας το **Adjusted Rand Index (ARI ≈ 0.37 - 0.38)**, παρατηρούμε μια μέτρια συσχέτιση με τις πραγματικές κλάσεις. "
             "Στην πράξη, επειδή το dataset αποτελείται κατά ~83% από κανονική κίνηση (BENIGN) και ~17% από επιθέσεις, οι δύο "
             "συσταδούλες που σχηματίζονται διαχωρίζουν σε μεγάλο βαθμό την κανονική κίνηση από τις επιθέσεις (δυαδικός διαχωρισμός), "
             "αλλά αποτυγχάνουν να ξεχωρίσουν τις 15 επιμέρους λεπτομερείς κατηγορίες επιθέσεων, καθώς πολλές επιθέσεις παρουσιάζουν "
             "παρόμοια δικτυακή συμπεριφορά.\n")
    rf.write("* **DBSCAN:** Ο DBSCAN, λόγω της πυκνοτικής του φύσης, καταφέρνει να εντοπίσει πολλές μικρές, "
             "συμπαγείς συστάδες δικτυακής κίνησης (π.χ. πολύ συγκεκριμένα είδη πακέτων ή επιθέσεων σάρωσης θυρών), "
             "ενώ απορρίπτει ως θόρυβο (outliers) τις μη-επαναλαμβανόμενες ανωμαλίες. Το χαμηλό ARI του DBSCAN "
             "οφείλεται στο ότι διασπά τις πραγματικές κλάσεις σε δεκάδες μικρότερα clusters, αλλά προσφέρει εξαιρετικά "
             "χαμηλό Davies-Bouldin Index (υψηλός τοπικός διαχωρισμός).\n\n")

    rf.write("### 5.3 Τελικά Συμπεράσματα\n")
    rf.write(f"* **K-Means:** Παραμένει ο πιο αποδοτικός αλγόριθμος για μεγάλα δεδομένα, δίνοντας μια καλή γενική "
             f"εικόνα του δυαδικού διαχωρισμού (BENIGN vs ATTACK).\n")
    rf.write(f"* **Ιεραρχική Ομαδοποίηση:** Η μέθοδος σύνδεσης **{best_hier_method}** παρήγαγε το καλύτερο "
             f"αποτέλεσμα, επιβεβαιώνοντας τη δομή δύο κύριων κλάδων στο δενδρόγραμμα.\n")
    rf.write(f"* **DBSCAN:** Αποδεικνύεται πολύτιμος για τον εντοπισμό ανώμαλης συμπεριφοράς (outliers/θόρυβος) "
             f"και για την ανακάλυψη μικρών, εξαιρετικά πυκνών τύπων κίνησης, χωρίς την ανάγκη εκ των προτέρων "
             f"γνώσης του αριθμού των συστάδων.\n")

print("Η αναφορά Report_Q3.md δημιουργήθηκε με επιτυχία.")
print("=== Ερώτημα 3 ολοκληρώθηκε με επιτυχία! ===")
