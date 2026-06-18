import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Reconfigure stdout to use UTF-8 for Greek Windows console
sys.stdout.reconfigure(encoding='utf-8')

output_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = output_dir
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

files = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]

print("=== Βήμα 1: Φόρτωση, Καθαρισμός και Συγχώνευση Δεδομένων ===")
dfs = []
total_raw_rows = 0

for f in files:
    path = os.path.join(dataset_dir, f)
    print(f"Ανάγνωση αρχείου: {f}...")
    df = pd.read_csv(path)
    total_raw_rows += len(df)
    
    # Καθαρισμός κενών στα ονόματα των στηλών
    df.columns = df.columns.str.strip()
    
    # Καθαρισμός τιμών της στήλης Label από προβληματικούς Unicode χαρακτήρες
    if 'Label' in df.columns:
        df['Label'] = df['Label'].str.replace('\ufffd', '-', regex=False).str.strip()
        
    # Προσωρινός καθαρισμός διπλοτύπων ανά αρχείο για εξοικονόμηση μνήμης
    df.drop_duplicates(inplace=True)
    
    # Διαχείριση NaN και Inf

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Αφαίρεση αρνητικών τιμών σε στήλες που πρέπει να είναι αυστηρά θετικές
    strictly_positive_cols = [
        'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s', 
        'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min'
    ]
    for col in strictly_positive_cols:
        if col in df.columns:
            df = df[df[col] >= 0]
    
    dfs.append(df)

print("Συγχώνευση όλων των αρχείων σε ένα ενιαίο DataFrame...")
full_df = pd.concat(dfs, ignore_index=True)

# Τελική αφαίρεση διπλοτύπων σε όλο το dataset
print("Αφαίρεση διπλοτύπων σε όλο το συγχωνευμένο dataset...")
full_df.drop_duplicates(inplace=True)

total_clean_rows = len(full_df)
print(f"Αρχικές γραμμές: {total_raw_rows}")
print(f"Καθαρές γραμμές μετά την αφαίρεση διπλοτύπων, NaN και Inf: {total_clean_rows}")
print(f"Αφαιρέθηκαν συνολικά: {total_raw_rows - total_clean_rows} γραμμές ({(total_raw_rows - total_clean_rows)/total_raw_rows*100:.2f}%)")

print("\n=== Βήμα 2: Στρωματοποιημένη Δειγματοληψία (Stratified Sampling) ===")
# Ορισμός συνάρτησης για δειγματοληψία με διατήρηση σπάνιων κλάσεων
def stratified_sample(df, target_col, sample_size=100000, min_samples=25, random_state=42):
    counts = df[target_col].value_counts()
    
    # Κλάσεις με πολύ λίγα δείγματα τις κρατάμε ολόκληρες
    rare_classes = counts[counts < min_samples].index.tolist()
    rare_df = df[df[target_col].isin(rare_classes)]
    remaining_df = df[~df[target_col].isin(rare_classes)]
    
    needed_sample_size = sample_size - len(rare_df)
    if needed_sample_size <= 0:
        return rare_df.sample(n=sample_size, random_state=random_state)
    
    # Στρωματοποιημένη δειγματοληψία στις υπόλοιπες κλάσεις
    _, sampled_remaining = train_test_split(
        remaining_df, 
        test_size=needed_sample_size, 
        stratify=remaining_df[target_col], 
        random_state=random_state
    )
    
    return pd.concat([rare_df, sampled_remaining], ignore_index=True)

sample_df = stratified_sample(full_df, 'Label', sample_size=100000, min_samples=50, random_state=42)
print(f"Μέγεθος δείγματος: {sample_df.shape}")

print("\n=== Βήμα 3: Ανάλυση Μεταβλητής Στόχου (Label) ===")
label_counts_full = full_df['Label'].value_counts()
label_counts_sample = sample_df['Label'].value_counts()

print("Κατανομή κλάσεων στο πλήρες dataset έναντι του δείγματος:")
for label in label_counts_full.index:
    full_count = label_counts_full[label]
    sample_count = label_counts_sample.get(label, 0)
    print(f"  {label:28} -> Πλήρες: {full_count:7d} ({full_count/total_clean_rows*100:6.3f}%) | Δείγμα: {sample_count:5d} ({sample_count/100000*100:6.3f}%)")

# Οπτικοποίηση κατανομής κλάσεων (Log Scale λόγω μεγάλης ανισορροπίας)
plt.figure(figsize=(12, 6))
sns.barplot(x=label_counts_sample.values, y=label_counts_sample.index, hue=label_counts_sample.index, palette="viridis", legend=False)
plt.xscale('log')
plt.title("Κατανομή Μεταβλητής Στόχου (Label) στο Δείγμα (Λογαριθμική Κλίμακα)")
plt.xlabel("Πλήθος Δειγμάτων (Log scale)")
plt.ylabel("Κατηγορία Κίνησης / Επίθεσης")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "label_distribution.png"), dpi=150)
plt.close()

print("\n=== Βήμα 4: Επιλογή Χαρακτηριστικών (Feature Selection) ===")
# 1. Αφαίρεση σταθερών στηλών (μηδενική διακύμανση)
X_sample = sample_df.drop(columns=['Label'])
constant_features = [col for col in X_sample.columns if X_sample[col].nunique() <= 1]
print(f"Σταθερές στήλες που αφαιρούνται (Μηδενική Διακύμανση): {constant_features}")

X_no_const = X_sample.drop(columns=constant_features)

# 2. Αφαίρεση διπλότυπων στηλών (με βάση τις τιμές)
duplicate_features = []
cols = X_no_const.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        col1 = cols[i]
        col2 = cols[j]
        if X_no_const[col1].equals(X_no_const[col2]):
            duplicate_features.append(col2)
duplicate_features = list(set(duplicate_features))
print(f"Διπλότυπες στήλες που αφαιρούνται: {duplicate_features}")

X_no_dup = X_no_const.drop(columns=duplicate_features)

# 3. Αφαίρεση στηλών με πολύ υψηλή συσχέτιση (Pearson Correlation > 0.95)
corr_matrix = X_no_dup.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop_corr = []
correlated_pairs = []

for column in upper_tri.columns:
    high_corr_features = upper_tri.index[upper_tri[column] > 0.95].tolist()
    if high_corr_features:
        to_drop_corr.append(column)
        for hcf in high_corr_features:
            correlated_pairs.append((column, hcf, corr_matrix.loc[column, hcf]))

print(f"Στήλες που αφαιρούνται λόγω υψηλής συσχέτισης (>0.95): {len(to_drop_corr)} στήλες")
for col, ref_col, val in correlated_pairs[:10]:
    print(f"  - '{col}' συσχετίζεται με '{ref_col}' με r = {val:.4f}")
if len(correlated_pairs) > 10:
    print(f"  ... και άλλες {len(correlated_pairs) - 10} συσχετίσεις.")

X_final = X_no_dup.drop(columns=to_drop_corr)
final_features = X_final.columns.tolist()
print(f"Τελικός αριθμός χαρακτηριστικών μετά την επιλογή: {len(final_features)}")

# Δημιουργία τελικού καθαρισμένου δείγματος
cleaned_sampled_df = pd.concat([X_final, sample_df['Label']], axis=1)
cleaned_sampled_df.to_csv(os.path.join(output_dir, "cleaned_sampled_data.csv"), index=False)
print("Το καθαρισμένο δείγμα αποθηκεύτηκε στο 'cleaned_sampled_data.csv'.")

print("\n=== Βήμα 5: Παραγωγή Γραφικών Παραστάσεων (Plots) ===")
# 1. Heatmap Συσχετίσεων για τα τελικά χαρακτηριστικά (αν είναι πολλά, παίρνουμε τα πρώτα 25)
features_to_plot = final_features[:25]
plt.figure(figsize=(16, 12))
sns.heatmap(X_final[features_to_plot].corr(), annot=False, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5)
plt.title("Heatmap Συσχέτισης Pearson για 25 Επιλεγμένα Χαρακτηριστικά")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"), dpi=150)
plt.close()

# 2. Box plots για επιλεγμένα χαρακτηριστικά (Outlier Detection)
key_numeric_cols = [
    'Flow Duration', 
    'Flow Bytes/s', 
    'Flow Packets/s', 
    'Packet Length Mean', 
    'Fwd Packet Length Mean', 
    'Bwd Packet Length Mean'
]
# Φιλτράρισμα για να κρατήσουμε μόνο όσα υπάρχουν στα τελικά χαρακτηριστικά
key_numeric_cols = [c for c in key_numeric_cols if c in final_features]

fig, axes = plt.subplots(len(key_numeric_cols), 1, figsize=(10, 2.5 * len(key_numeric_cols)))
if len(key_numeric_cols) == 1:
    axes = [axes]

for i, col in enumerate(key_numeric_cols):
    sns.boxplot(x=cleaned_sampled_df[col], ax=axes[i])
    axes[i].set_title(f"Boxplot για το χαρακτηριστικό: {col}")
    axes[i].set_xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "boxplots_key_features.png"), dpi=150)
plt.close()

# Υπολογισμός στατιστικών για τα τελικά χαρακτηριστικά
stats_df = X_final.describe().T
stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

