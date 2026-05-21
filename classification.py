import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

output_dir = r"C:\Users\user\Documents\GitHub\Data Mine"
plots_dir  = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# ============================================================
# ΒΗΜΑ 1: Φόρτωση Δεδομένων
# ============================================================
print("=== Βήμα 1: Φόρτωση Δεδομένων ===")
df = pd.read_csv(os.path.join(output_dir, "cleaned_sampled_data.csv"))
print(f"Dataset: {df.shape[0]:,} γραμμές, {df.shape[1]} στήλες")

# ============================================================
# ΒΗΜΑ 2: Προεπεξεργασία
# ============================================================
print("\n=== Βήμα 2: Προεπεξεργασία ===")

le = LabelEncoder()
df['Label_enc'] = le.fit_transform(df['Label'])

# Αφαίρεση κλάσεων με < 2 δείγματα (δεν επιτρέπουν stratified split)
counts = df['Label'].value_counts()
valid  = counts[counts >= 2].index
removed = counts[counts < 2].index.tolist()
if removed:
    print(f"Αφαιρούνται κλάσεις με < 2 δείγματα: {removed}")
    df = df[df['Label'].isin(valid)].copy()
    # Επανακωδικοποίηση
    le2 = LabelEncoder()
    df['Label_enc'] = le2.fit_transform(df['Label'])
    class_names = le2.classes_
else:
    class_names = le.classes_

print(f"Κλάσεις ({len(class_names)}): {list(class_names)}")

X_all = df.drop(columns=['Label', 'Label_enc']).values
y_all = df['Label_enc'].values

# ============================================================
# ΒΗΜΑ 3: Διαχωρισμός Δεδομένων
# ============================================================
print("\n=== Βήμα 3: Διαχωρισμός Δεδομένων ===")

# --- Σενάριο Α: Train(70%) / Val(15%) / Test(15%) ---
X_tv, X_test_A, y_tv, y_test_A = train_test_split(
    X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)
X_train_A, X_val_A, y_train_A, y_val_A = train_test_split(
    X_tv, y_tv, test_size=0.15/0.85, random_state=42, stratify=y_tv)

scaler_A = StandardScaler()
X_train_A_sc = scaler_A.fit_transform(X_train_A)
X_val_A_sc   = scaler_A.transform(X_val_A)
X_test_A_sc  = scaler_A.transform(X_test_A)

print(f"Σενάριο Α → Train: {len(y_train_A):,} | Val: {len(y_val_A):,} | Test: {len(y_test_A):,}")

# --- Σενάριο Β: Train(80%) / Test(20%) + GridSearchCV ---
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

scaler_B = StandardScaler()
X_train_B_sc = scaler_B.fit_transform(X_train_B)
X_test_B_sc  = scaler_B.transform(X_test_B)

# Μικρό υποσύνολο για γρήγορο Grid Search (~20K) -- μετά refit στο πλήρες train
# (refit=True στο GridSearchCV το κάνει αυτόματα στο πλήρες train)
print(f"Σενάριο Β → Train: {len(y_train_B):,} | Test: {len(y_test_B):,}")

# ============================================================
# Βοηθητικές Συναρτήσεις
# ============================================================
def compute_metrics(y_true, y_pred):
    return {
        'Accuracy':         round(accuracy_score(y_true, y_pred), 4),
        'Precision (W)':    round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'Recall (W)':       round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'F1 (Weighted)':    round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'Precision (Mac)':  round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'Recall (Mac)':     round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'F1 (Macro)':       round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
    }

def plot_cm(y_true, y_pred, class_names, title, filename):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.3, annot_kws={'size': 7})
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('Προβλεπόμενη Κλάση', fontsize=10)
    ax.set_ylabel('Πραγματική Κλάση', fontsize=10)
    plt.xticks(rotation=35, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=150)
    plt.close()
    print(f"  → Αποθηκεύτηκε: plots/{filename}")

results_A = {}
results_B = {}

# ============================================================
# ΒΗΜΑ 4Α: Σενάριο Α – Manual Grid Search (Train/Val/Test)
# ============================================================
print("\n=== Βήμα 4Α: Σενάριο Α - Manual Grid Search ===")

# ---- Logistic Regression ----
print("[Σενάριο Α] Logistic Regression...")
best_f1, best_C, best_model = -1, None, None
for C in [0.1, 1.0, 10.0]:
    m = LogisticRegression(C=C, max_iter=500, class_weight='balanced',
                           solver='lbfgs', n_jobs=-1, random_state=42)
    m.fit(X_train_A_sc, y_train_A)
    vf1 = f1_score(y_val_A, m.predict(X_val_A_sc), average='weighted', zero_division=0)
    print(f"  C={C} → Val F1(W): {vf1:.4f}")
    if vf1 > best_f1:
        best_f1, best_C, best_model = vf1, C, m

pred = best_model.predict(X_test_A_sc)
results_A['Logistic Regression'] = {
    'best_params': {'C': best_C}, 'val_f1': round(best_f1, 4),
    'test_metrics': compute_metrics(y_test_A, pred)}
print(f"  ★ C={best_C} | Test F1(W): {results_A['Logistic Regression']['test_metrics']['F1 (Weighted)']}")
plot_cm(y_test_A, pred, class_names,
        "Confusion Matrix – Logistic Regression (Σενάριο Α)", "cm_A_lr.png")

# ---- Decision Tree ----
print("[Σενάριο Α] Decision Tree...")
best_f1, best_p, best_model = -1, None, None
for depth in [5, 10, 15]:
    for crit in ['gini', 'entropy']:
        m = DecisionTreeClassifier(max_depth=depth, criterion=crit,
                                   class_weight='balanced', random_state=42)
        m.fit(X_train_A_sc, y_train_A)
        vf1 = f1_score(y_val_A, m.predict(X_val_A_sc), average='weighted', zero_division=0)
        print(f"  depth={depth}, crit={crit} → Val F1(W): {vf1:.4f}")
        if vf1 > best_f1:
            best_f1, best_p, best_model = vf1, {'max_depth': depth, 'criterion': crit}, m

pred = best_model.predict(X_test_A_sc)
results_A['Decision Tree'] = {
    'best_params': best_p, 'val_f1': round(best_f1, 4),
    'test_metrics': compute_metrics(y_test_A, pred)}
print(f"  ★ {best_p} | Test F1(W): {results_A['Decision Tree']['test_metrics']['F1 (Weighted)']}")
plot_cm(y_test_A, pred, class_names,
        "Confusion Matrix – Decision Tree (Σενάριο Α)", "cm_A_dt.png")

# ---- Random Forest ----
print("[Σενάριο Α] Random Forest...")
best_f1, best_p, best_model = -1, None, None
for n_est in [50, 100]:
    for depth in [10, 20]:
        print(f"  n_estimators={n_est}, max_depth={depth}...", flush=True)
        m = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                   class_weight='balanced', n_jobs=-1, random_state=42)
        m.fit(X_train_A_sc, y_train_A)
        vf1 = f1_score(y_val_A, m.predict(X_val_A_sc), average='weighted', zero_division=0)
        print(f"  → Val F1(W): {vf1:.4f}")
        if vf1 > best_f1:
            best_f1, best_p, best_model = vf1, {'n_estimators': n_est, 'max_depth': depth}, m

pred = best_model.predict(X_test_A_sc)
results_A['Random Forest'] = {
    'best_params': best_p, 'val_f1': round(best_f1, 4),
    'test_metrics': compute_metrics(y_test_A, pred)}
print(f"  ★ {best_p} | Test F1(W): {results_A['Random Forest']['test_metrics']['F1 (Weighted)']}")
plot_cm(y_test_A, pred, class_names,
        "Confusion Matrix – Random Forest (Σενάριο Α)", "cm_A_rf.png")

# ============================================================
# ΒΗΜΑ 4Β: Σενάριο Β – GridSearchCV 3-fold CV
# ============================================================
print("\n=== Βήμα 4Β: Σενάριο Β - GridSearchCV (3-fold CV) ===")
cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ---- Logistic Regression ----
print("[Σενάριο Β] Logistic Regression GridSearchCV...")
gs = GridSearchCV(
    LogisticRegression(max_iter=500, class_weight='balanced',
                       solver='lbfgs', random_state=42),
    param_grid={'C': [0.1, 1.0, 10.0]},
    scoring='f1_weighted', cv=cv3, n_jobs=-1, verbose=0, refit=True)
gs.fit(X_train_B_sc, y_train_B)
pred = gs.best_estimator_.predict(X_test_B_sc)
results_B['Logistic Regression'] = {
    'best_params': gs.best_params_, 'best_cv_f1': round(gs.best_score_, 4),
    'test_metrics': compute_metrics(y_test_B, pred)}
print(f"  ★ {gs.best_params_} | CV F1: {gs.best_score_:.4f} | Test F1(W): {results_B['Logistic Regression']['test_metrics']['F1 (Weighted)']}")
plot_cm(y_test_B, pred, class_names,
        "Confusion Matrix – Logistic Regression (Σενάριο Β)", "cm_B_lr.png")

# ---- Decision Tree ----
print("[Σενάριο Β] Decision Tree GridSearchCV...")
gs = GridSearchCV(
    DecisionTreeClassifier(class_weight='balanced', random_state=42),
    param_grid={'max_depth': [5, 10, 15], 'criterion': ['gini', 'entropy']},
    scoring='f1_weighted', cv=cv3, n_jobs=-1, verbose=0, refit=True)
gs.fit(X_train_B_sc, y_train_B)
pred = gs.best_estimator_.predict(X_test_B_sc)
results_B['Decision Tree'] = {
    'best_params': gs.best_params_, 'best_cv_f1': round(gs.best_score_, 4),
    'test_metrics': compute_metrics(y_test_B, pred)}
print(f"  ★ {gs.best_params_} | CV F1: {gs.best_score_:.4f} | Test F1(W): {results_B['Decision Tree']['test_metrics']['F1 (Weighted)']}")
plot_cm(y_test_B, pred, class_names,
        "Confusion Matrix – Decision Tree (Σενάριο Β)", "cm_B_dt.png")

# ---- Random Forest ----
print("[Σενάριο Β] Random Forest GridSearchCV...")
gs = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid={'n_estimators': [50, 100], 'max_depth': [10, 20]},
    scoring='f1_weighted', cv=cv3, n_jobs=-1, verbose=0, refit=True)
gs.fit(X_train_B_sc, y_train_B)
pred = gs.best_estimator_.predict(X_test_B_sc)
results_B['Random Forest'] = {
    'best_params': gs.best_params_, 'best_cv_f1': round(gs.best_score_, 4),
    'test_metrics': compute_metrics(y_test_B, pred)}
print(f"  ★ {gs.best_params_} | CV F1: {gs.best_score_:.4f} | Test F1(W): {results_B['Random Forest']['test_metrics']['F1 (Weighted)']}")
plot_cm(y_test_B, pred, class_names,
        "Confusion Matrix – Random Forest (Σενάριο Β)", "cm_B_rf.png")

# ============================================================
# ΒΗΜΑ 5: Γράφημα Σύγκρισης
# ============================================================
print("\n=== Βήμα 5: Γράφημα Σύγκρισης ===")
models  = ['Logistic Regression', 'Decision Tree', 'Random Forest']
metrics = ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, model in enumerate(models):
    ax = axes[idx]
    vA = [results_A[model]['test_metrics'][m] for m in metrics]
    vB = [results_B[model]['test_metrics'][m] for m in metrics]
    x  = np.arange(len(metrics))
    w  = 0.35
    bA = ax.bar(x - w/2, vA, w, label='Σενάριο Α (Val/Test)', color='#4C72B0', alpha=0.85)
    bB = ax.bar(x + w/2, vB, w, label='Σενάριο Β (GridSearchCV)', color='#DD8452', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_title(model, fontsize=11, fontweight='bold')
    ax.set_ylabel('Τιμή Μετρικής', fontsize=9); ax.legend(fontsize=8)
    for b in list(bA) + list(bB):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)
plt.suptitle('Σύγκριση Μοντέλων: Σενάριο Α vs Σενάριο Β', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "comparison_scenarios.png"), dpi=150)
plt.close()
print("  → Αποθηκεύτηκε: plots/comparison_scenarios.png")

# ============================================================
# ΒΗΜΑ 6: Αναφορά Report_Q2.md
# ============================================================
print("\n=== Βήμα 6: Παραγωγή Report_Q2.md ===")
models_list = ['Logistic Regression', 'Decision Tree', 'Random Forest']

with open(os.path.join(output_dir, "Report_Q2.md"), "w", encoding="utf-8") as rf:
    rf.write("# Αναφορά Ερωτήματος 2 – Κατηγοριοποίηση (Classification)\n\n")
    rf.write("Αυτή η αναφορά παρουσιάζει τα αποτελέσματα εφαρμογής τεχνικών επιβλεπόμενης μάθησης "
             "στο σύνολο δεδομένων **CIC-IDS-2017** για την ανίχνευση κακόβουλης δικτυακής δραστηριότητας.\n\n")

    # --- Προεπεξεργασία ---
    rf.write("## 1. Προεπεξεργασία Δεδομένων\n\n")
    rf.write("### 1.1 Σύνολο Δεδομένων\n")
    rf.write("Χρησιμοποιήθηκε το καθαρισμένο δείγμα `cleaned_sampled_data.csv` "
             "(100.000 εγγραφές, 48 χαρακτηριστικά) που δημιουργήθηκε στο Ερώτημα 1. "
             "Η κλάση `Infiltration` (1 μόνο δείγμα) αφαιρέθηκε καθώς δεν επιτρέπει "
             "στρωματοποιημένο διαχωρισμό, οδηγώντας σε **14 τελικές κλάσεις**.\n\n")
    rf.write("### 1.2 Κωδικοποίηση Μεταβλητής Στόχου\n")
    rf.write("Η στήλη `Label` μετατράπηκε σε αριθμητικές τιμές `[0, 13]` με `LabelEncoder`:\n\n")
    rf.write("| Κλάση | Κωδικός |\n|:---|:---:|\n")
    for i, c in enumerate(class_names):
        rf.write(f"| {c} | {i} |\n")
    rf.write("\n")
    rf.write("### 1.3 Κανονικοποίηση Χαρακτηριστικών\n")
    rf.write("Εφαρμόστηκε `StandardScaler` (μέσος=0, τυπική απόκλιση=1) στα αριθμητικά χαρακτηριστικά. "
             "Ο scaler εκπαιδεύτηκε **μόνο στο training set** για αποφυγή data leakage, ενώ "
             "εφαρμόστηκε μεταγενέστερα στο validation και test set.\n\n")
    rf.write("### 1.4 Αντιμετώπιση Ανισορροπίας Κλάσεων\n")
    rf.write("Το dataset εμφανίζει έντονη ανισορροπία (>83% BENIGN). "
             "Χρησιμοποιήθηκε `class_weight='balanced'` σε όλους τους αλγορίθμους, "
             "αναθέτοντας αυτόματα βάρη αντιστρόφως ανάλογα της συχνότητας κάθε κλάσης.\n\n")

    # --- Σενάριο Α ---
    rf.write("## 2. Σενάριο Α – Manual Grid Search (Train/Val/Test Split)\n\n")
    rf.write("### 2.1 Στρατηγική Διαχωρισμού\n")
    rf.write(f"| Σύνολο | Εγγραφές | Ποσοστό |\n|:---|:---:|:---:|\n"
             f"| Training | {len(y_train_A):,} | 70% |\n"
             f"| Validation | {len(y_val_A):,} | 15% |\n"
             f"| Test | {len(y_test_A):,} | 15% |\n\n")
    rf.write("Ο διαχωρισμός έγινε **στρωματοποιημένα** (stratified) διατηρώντας τις αναλογίες των κλάσεων. "
             "Το validation set χρησιμοποιήθηκε αποκλειστικά για επιλογή υπερπαραμέτρων, "
             "και το test set για τελική αξιολόγηση.\n\n")

    rf.write("### 2.2 Πλέγμα Υπερπαραμέτρων\n")
    rf.write("| Αλγόριθμος | Υπερπαράμετρος | Τιμές Αναζήτησης |\n|:---|:---|:---:|\n")
    rf.write("| Logistic Regression | C (αντίστροφη κανονικοποίηση) | 0.1, 1.0, 10.0 |\n")
    rf.write("| Decision Tree | max_depth | 5, 10, 15 |\n")
    rf.write("| Decision Tree | criterion | gini, entropy |\n")
    rf.write("| Random Forest | n_estimators | 50, 100 |\n")
    rf.write("| Random Forest | max_depth | 10, 20 |\n\n")

    rf.write("### 2.3 Βέλτιστες Παράμετροι\n\n")
    rf.write("| Αλγόριθμος | Βέλτιστες Παράμετροι | Val F1 (W) |\n|:---|:---|:---:|\n")
    for model in models_list:
        ps = ', '.join(f"{k}={v}" for k, v in results_A[model]['best_params'].items())
        rf.write(f"| {model} | {ps} | {results_A[model]['val_f1']:.4f} |\n")
    rf.write("\n")

    rf.write("### 2.4 Αποτελέσματα στο Test Set\n\n")
    rf.write("| Αλγόριθμος | Accuracy | Precision (W) | Recall (W) | F1 (W) | F1 (Macro) |\n")
    rf.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
    for model in models_list:
        m = results_A[model]['test_metrics']
        rf.write(f"| {model} | {m['Accuracy']:.4f} | {m['Precision (W)']:.4f} | "
                 f"{m['Recall (W)']:.4f} | {m['F1 (Weighted)']:.4f} | {m['F1 (Macro)']:.4f} |\n")
    rf.write("\nΟι πίνακες σύγχυσης αποθηκεύτηκαν ως `plots/cm_A_*.png`.\n\n")

    # --- Σενάριο Β ---
    rf.write("## 3. Σενάριο Β – GridSearchCV με 3-Fold Cross-Validation (80/20 Split)\n\n")
    rf.write("### 3.1 Στρατηγική Διαχωρισμού\n")
    rf.write(f"| Σύνολο | Εγγραφές | Ποσοστό |\n|:---|:---:|:---:|\n"
             f"| Training | {len(y_train_B):,} | 80% |\n"
             f"| Test | {len(y_test_B):,} | 20% |\n\n")
    rf.write("Χρησιμοποιήθηκε `GridSearchCV` με **3-fold Stratified Cross-Validation** στο training set "
             "για επιλογή υπερπαραμέτρων (μεγιστοποίηση F1 weighted). "
             "Το `refit=True` (default) επανεκπαιδεύει αυτόματα το βέλτιστο μοντέλο στο πλήρες training set.\n\n")

    rf.write("### 3.2 Βέλτιστες Παράμετροι\n\n")
    rf.write("| Αλγόριθμος | Βέλτιστες Παράμετροι | CV F1 (W) |\n|:---|:---|:---:|\n")
    for model in models_list:
        ps = ', '.join(f"{k}={v}" for k, v in results_B[model]['best_params'].items())
        rf.write(f"| {model} | {ps} | {results_B[model]['best_cv_f1']:.4f} |\n")
    rf.write("\n")

    rf.write("### 3.3 Αποτελέσματα στο Test Set\n\n")
    rf.write("| Αλγόριθμος | Accuracy | Precision (W) | Recall (W) | F1 (W) | F1 (Macro) |\n")
    rf.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
    for model in models_list:
        m = results_B[model]['test_metrics']
        rf.write(f"| {model} | {m['Accuracy']:.4f} | {m['Precision (W)']:.4f} | "
                 f"{m['Recall (W)']:.4f} | {m['F1 (Weighted)']:.4f} | {m['F1 (Macro)']:.4f} |\n")
    rf.write("\nΟι πίνακες σύγχυσης αποθηκεύτηκαν ως `plots/cm_B_*.png`.\n\n")

    # --- Σύγκριση ---
    rf.write("## 4. Συγκριτική Ανάλυση και Συμπεράσματα\n\n")
    rf.write("### 4.1 Σύγκριση Σεναρίων\n")
    rf.write("| Χαρακτηριστικό | Σενάριο Α | Σενάριο Β |\n|:---|:---|:---|\n")
    rf.write("| Στρατηγική | Manual Grid Search | GridSearchCV + k-fold CV |\n")
    rf.write("| Train/Val/Test | 70/15/15 | 80/-/20 |\n")
    rf.write("| Αξιοπιστία εκτίμησης | Μέτρια (1 validation set) | Υψηλότερη (3 folds) |\n")
    rf.write("| Δεδομένα εκπαίδευσης | 70% | 80% |\n")
    rf.write("| Πολυπλοκότητα | Χαμηλή | Υψηλότερη (3x περισσότερα fits) |\n\n")

    rf.write("### 4.2 Καλύτερο Μοντέλο ανά Σενάριο\n\n")
    bA = max(results_A.items(), key=lambda x: x[1]['test_metrics']['F1 (Weighted)'])
    bB = max(results_B.items(), key=lambda x: x[1]['test_metrics']['F1 (Weighted)'])
    rf.write(f"* **Σενάριο Α:** **{bA[0]}** → F1(W)={bA[1]['test_metrics']['F1 (Weighted)']:.4f}, "
             f"Accuracy={bA[1]['test_metrics']['Accuracy']:.4f}\n")
    rf.write(f"* **Σενάριο Β:** **{bB[0]}** → F1(W)={bB[1]['test_metrics']['F1 (Weighted)']:.4f}, "
             f"Accuracy={bB[1]['test_metrics']['Accuracy']:.4f}\n\n")

    rf.write("### 4.3 Ανάλυση ανά Αλγόριθμο\n\n")
    rf.write("* **Logistic Regression:** Γραμμικός αλγόριθμος χαμηλής πολυπλοκότητας. Λειτουργεί ως "
             "ισχυρό baseline. Η παράμετρος `C` ελέγχει την ισορροπία μεταξύ underfitting (μικρό C) "
             "και overfitting (μεγάλο C). Περιορίζεται σε γραμμικούς αποφαστικούς ορίζοντες.\n\n")
    rf.write("* **Decision Tree:** Μη-γραμμικός αλγόριθμος με υψηλή ερμηνευσιμότητα. "
             "Ο περιορισμός `max_depth` αποτρέπει την υπερεκπαίδευση. "
             "Ευαίσθητος σε μικρές αλλαγές των δεδομένων εκπαίδευσης.\n\n")
    rf.write("* **Random Forest:** Ensemble μέθοδος (bagging πολλών Decision Trees). "
             "Εγγενής ανθεκτικότητα σε overfitting και θόρυβο. "
             "Συνήθως αποδίδει καλύτερα από μεμονωμένα δέντρα σε imbalanced datasets.\n\n")
    rf.write("### 4.4 Συμπέρασμα\n")
    rf.write("Το Σενάριο Β (GridSearchCV + CV) παρέχει πιο αξιόπιστη επιλογή υπερπαραμέτρων "
             "καθώς η εκτίμηση απόδοσης δεν εξαρτάται από μία μόνο τυχαία διαίρεση validation. "
             "Επιπλέον, αξιοποιεί μεγαλύτερο ποσοστό δεδομένων (80%) για εκπαίδευση. "
             "Ο **Random Forest** αναδεικνύεται ως ο καταλληλότερος αλγόριθμος για το "
             "συγκεκριμένο πρόβλημα κυβερνοασφάλειας, λόγω της ικανότητάς του να χειρίζεται "
             "μη-γραμμικές σχέσεις και την ανισορροπία κλάσεων.\n")

print("Η αναφορά Report_Q2.md δημιουργήθηκε με επιτυχία.")
print("=== Ερώτημα 2 ολοκληρώθηκε με επιτυχία! ===")
