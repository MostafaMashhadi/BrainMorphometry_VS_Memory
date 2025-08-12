import pandas as pd
import numpy as np
from scipy.stats import pearsonr, linregress
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# ------------ Paths and output folder ------------
# Assuming this script is in `src/`, the outputs folder will be created one level above it
base_path = "/Users/mostafamashhadizadeh/Desktop/MyProjects/BrainMorphometry_MemoryStudy/data/"
out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(out_dir, exist_ok=True)

# ------------ Read input data ------------
vmri = pd.read_csv(base_path + "A4_VMRI_PRV2_26Jul2025.csv")
pet_amyloid = pd.read_csv(base_path + "A4_PETSUVR_PRV2_26Jul2025.csv")
pet_tau = pd.read_csv(base_path + "TAUSUVR_PETSURFER_26Jul2025.csv")

outcome_files = [
    {"file": "A4_COGDIGIT_PRV2_26Jul2025.csv", "col": "DIGITTOTAL"},
    {"file": "A4_COGLOGIC_PRV2_26Jul2025.csv", "col": "LDELTOTAL"},
    {"file": "A4_COGLOGIC_PRV2_26Jul2025.csv", "col": "LIMMTOTAL"},
    {"file": "A4_CDR_PRV2_26Jul2025.csv", "col": "CDSOB"},
]

# --- Merge outcomes ---
outcomes_df = None
for info in outcome_files:
    df = pd.read_csv(base_path + info["file"])[["BID", info["col"]]]
    outcomes_df = df if outcomes_df is None else pd.merge(outcomes_df, df, on="BID", how="outer")

# --- Extract MRI features and normalize by ICV ---
exclude_cols = ["BID", "VISCODE", "update_stamp", "IntraCranialVolume"]
mri_feats = [c for c in vmri.columns if c not in exclude_cols and np.issubdtype(vmri[c].dtype, np.number)]

vmri["IntraCranialVolume"] = vmri["IntraCranialVolume"].replace(0, np.nan)
vmri_raw = vmri.copy()
for col in mri_feats:
    vmri[col] = vmri[col] / vmri["IntraCranialVolume"]

# --- PET Amyloid composite ---
pet_amyloid_composite = pet_amyloid[pet_amyloid["brain_region"] == "Composite_Summary"]
pet_amyloid_df = pet_amyloid_composite.pivot_table(index="BID", columns="brain_region", values="suvr_cer").reset_index()

# --- PET Tau ---
tau_feats = [c for c in pet_tau.columns if c.startswith("bi_") and np.issubdtype(pet_tau[c].dtype, np.number)]
pet_tau_df = pet_tau[["ID"] + tau_feats].rename(columns={"ID": "BID"})

# --- Merge all to single dataframe ---
df_all = outcomes_df.merge(vmri[["BID"] + mri_feats + ["IntraCranialVolume"]], on="BID", how="inner")
df_all = df_all.merge(pet_amyloid_df, on="BID", how="inner")
df_all = df_all.merge(pet_tau_df, on="BID", how="inner")

# --- Filter to remove rows without any outcome values ---
outcome_cols = [x["col"] for x in outcome_files]
df_all = df_all.dropna(subset=outcome_cols, how="all")

# --- Helper function to compute correlation matrix ---
def corr_matrix(features, label_names):
    results = pd.DataFrame(index=label_names, columns=features, dtype=float)
    for outcome in label_names:
        sub = df_all[[outcome] + features].dropna(subset=[outcome])
        sub[features] = sub[features].fillna(sub[features].mean())
        for feat in features:
            if sub[feat].nunique() > 1:
                r, _ = pearsonr(sub[feat], sub[outcome])
                results.loc[outcome, feat] = r
    order = results.abs().max().sort_values(ascending=False).index
    return results[order]

mri_corr = corr_matrix(mri_feats, outcome_cols)
tau_corr = corr_matrix(tau_feats, outcome_cols)

sns.set(style="whitegrid", font_scale=1.4)

# --- Helper function to save figures ---
def save_fig(name):
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=400, bbox_inches="tight")
    plt.close()

# ------------ 1. Distributions (FacetGrid) ------------
dist_cols = ["DIGITTOTAL",
             "ForebrainParenchyma", "RightHippocampus", "HOC",
             "LeftInferiorLateralVentricle", "LeftLateralVentricle",
             "LeftSuperiortemporal", "LeftSuperiorparietal"]

df_long = df_all[dist_cols].melt(var_name="Feature", value_name="Value")
g = sns.FacetGrid(df_long, col="Feature", col_wrap=4, sharex=False, sharey=False, height=3)
g.map(sns.histplot, "Value", kde=True, color="steelblue")
g.set_titles("{col_name}")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Distributions of Selected MRI and Outcome Variables", fontsize=16)
save_fig("facet_distributions")

# ------------ 2. ICV normalization effect plot ------------
plt.figure(figsize=(6,5))
sns.scatterplot(x=vmri_raw["LeftHippocampus"], y=vmri["LeftHippocampus"], 
                hue=df_all["DIGITTOTAL"], palette="viridis", alpha=0.8)
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=plt.gca(), label="DIGITTOTAL")
cbar.ax.set_position([0.92, 0.25, 0.02, 0.6])
plt.xlabel("Left Hippocampus (Raw)")
plt.ylabel("Left Hippocampus (ICV-normalized)")
plt.title("ICV Normalization Effect", fontsize=16)
save_fig("icv_effect_hippocampus")

# ------------ 3. Heatmaps ------------
plt.figure(figsize=(16,6))
sns.heatmap(mri_corr.dropna(axis=1, how="all"), cmap="coolwarm", center=0, cbar_kws={"label": "Pearson r"})
plt.title("MRI Volumes vs Cognitive Outcomes", fontsize=18)
save_fig("heatmap_MRI_vs_outcomes")

plt.figure(figsize=(16,6))
sns.heatmap(tau_corr.dropna(axis=1, how="all"), cmap="coolwarm", center=0, cbar_kws={"label": "Pearson r"})
plt.title("Tau PET SUVR vs Cognitive Outcomes", fontsize=18)
save_fig("heatmap_Tau_vs_outcomes")

# ------------ 4. Barplots: top-10 MRI correlations ----------
for outcome in mri_corr.index:
    top_regions = mri_corr.loc[outcome].dropna()
    if None in top_regions.index:
        top_regions = top_regions.drop(index=None)
    top_regions = top_regions.abs().sort_values(ascending=False).head(10)
    plt.figure(figsize=(6,4))
    sns.barplot(x=top_regions, y=top_regions.index, hue=top_regions.index, palette="coolwarm", legend=False)
    plt.title(f"Top 10 MRI Regions correlated with {outcome}", fontsize=16)
    plt.ylabel("Brain Regins")
    plt.xlabel("Absolute Pearson r")
    save_fig(f"barplot_top10_MRI_{outcome}")

# ------------ 5. Scatter plots (selected regions) ------------
selected_regions = ["HOC", "RightHippocampus", "LeftSuperiorparietal"]

for region in selected_regions:
    plt.figure(figsize=(6,5))
    sc = plt.scatter(df_all[region], df_all["DIGITTOTAL"],
                     c=df_all["IntraCranialVolume"], cmap="coolwarm", alpha=0.8)
    cbar = plt.colorbar(sc, ax=plt.gca(), label="IntraCranialVolume")
    cbar.ax.set_position([0.92, 0.25, 0.02, 0.6])

    x, y = df_all[region], df_all["DIGITTOTAL"]
    mask = ~x.isna() & ~y.isna()
    if mask.sum() > 2:
        slope, intercept, r_val, p_val, _ = linregress(x[mask], y[mask])
        if p_val < 0.05:
            sns.regplot(x=x, y=y, scatter=False, color="black", line_kws={"lw":1})

    plt.title(f"{region} vs DIGITTOTAL", fontsize=16)
    plt.xlabel(region)
    plt.ylabel("DIGITTOTAL")
    save_fig(f"scatter_{region}_vs_DIGITTOTAL")

# ------------ 6. PCA embedding ----------
mri_data = df_all[mri_feats].fillna(df_all[mri_feats].mean())
scaler = StandardScaler()
mri_scaled = scaler.fit_transform(mri_data)
pca = PCA(n_components=2)
Z = pca.fit_transform(mri_scaled)
plt.figure(figsize=(6,5))
pc = plt.scatter(Z[:,0], Z[:,1], c=df_all["DIGITTOTAL"], cmap="viridis", alpha=0.8)
cbar = plt.colorbar(pc, ax=plt.gca(), label="DIGITTOTAL")
cbar.ax.set_position([0.92, 0.25, 0.02, 0.6])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of MRI features", fontsize=16)
save_fig("pca_mri_DIGITTOTAL")

# ------------ 7. UMAP embedding ----------
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
Z_umap = reducer.fit_transform(mri_scaled)
plt.figure(figsize=(6,5))
uc = plt.scatter(Z_umap[:,0], Z_umap[:,1], c=df_all["DIGITTOTAL"], cmap="viridis", alpha=0.8)
cbar = plt.colorbar(uc, ax=plt.gca(), label="DIGITTOTAL")
cbar.ax.set_position([0.92, 0.25, 0.02, 0.6])
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("UMAP of MRI features", fontsize=16)
save_fig("umap_mri_DIGITTOTAL")

print(f"All figures saved in {out_dir}")

