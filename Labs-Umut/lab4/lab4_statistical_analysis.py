"""
Lab 4 – Statistical Analysis (Full Script)
Path: labs/lab4/lab4_statistical_analysis.py

Bu script md'deki gereksinimler doğrultusunda:
- Veri yükleme (datasets/ ve /mnt/data yedek)
- Esnek sütun eşleme (concrete, loads, materials)
- Tanımlayıcı istatistikler (mean/median/mode/std/var/range/IQR/skew/kurtosis/percentiles)
- Görseller: histogram+KDE+işaretler, boxplot, loads dağılımı, fit overlay, QQ-plot, Bayes olasılık ağacı, mini dashboard
- Olasılıklar: Bernoulli, Binomial, Poisson, Uniform, Normal, Exponential
- Normal dağılım fit (MLE) + overlay + sentetik kıyas
- Bayes güncellemesi + probability tree
- Rapor: lab4_statistical_report.txt
- Bol # açıklama, PEP8'e yakın, hataya dayanıklı (try/except ve korumalar)

Gereksinimler:
pip install pandas numpy matplotlib seaborn scipy
"""

# ==== İTHALATLAR ==============================================================
import os  # # Yol işlemleri
import textwrap  # # Rapor satır kırpma
from typing import Dict, Optional, Tuple, List  # # Tip ipuçları

import numpy as np  # # Sayısal işlemler
import pandas as pd  # # DataFrame işlemleri
import seaborn as sns  # # İstatistiksel görsel
import matplotlib.pyplot as plt  # # Grafik
from scipy import stats  # # QQ-plot vs.
from scipy.stats import norm, binom, poisson, uniform, expon  # # Dağılımlar

# ==== GÖRSEL TEMA =============================================================
sns.set(style="whitegrid", context="notebook")  # # Temiz görünüm
plt.rcParams["figure.figsize"] = (9, 5.5)      # # Varsayılan figür boyutu
# NOT: bbox_inches bir rcParam değildir; savefig(..., bbox_inches="tight") ile kullanılır.


# ==== YOL TAHMİNİ =============================================================
def _guess_paths() -> Tuple[str, str, str]:
    """Script konumuna göre repo kökü ve datasets yolunu bulur; /mnt/data fallback."""
    here = os.path.abspath(os.path.dirname(__file__))            # labs/lab4
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))  # repo kökü
    datasets_root = os.path.join(repo_root, "datasets")          # datasets/
    fallback_root = "/mnt/data"                                  # sohbet ortamı
    return repo_root, datasets_root, fallback_root


REPO_ROOT, DATASETS_ROOT, FALLBACK_ROOT = _guess_paths()
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


# ==== ESNEK SÜTUN BULUCU ======================================================
def _normalize(name: str) -> str:
    """Sütun adını kıyaslama için normalize eder (küçült, boşluk->_, parantezleri sil)."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("[", "")
        .replace("]", "")
    )


def _find_column(candidates: List[str], df_columns: pd.Index) -> Optional[str]:
    """
    Birden fazla aday isim listesine göre veri içinden ilk uygun sütunu döndürür.
    - Tam listeyi geniş tuttuk; farklı dataset isimlendirmeleriyle uyumlu çalışır.
    """
    lower_map = {_normalize(c): c for c in df_columns}
    for cand in candidates:
        key = _normalize(cand)
        if key in lower_map:
            return lower_map[key]
    # Ek: kısmi yakalama (ör. 'strength' içeren)
    for cand in candidates:
        key = _normalize(cand)
        for col in df_columns:
            if key in _normalize(col):
                return col
    return None


# ==== YARDIMCI: DOSYA YÜKLEME =================================================
def load_data(file_name: str) -> pd.DataFrame:
    """
    CSV dosyasını datasets/ veya /mnt/data altından yükler.
    Yol adayları sırayla denenir; ilki başarılı olan kullanılır.
    """
    candidates = [
        os.path.join(DATASETS_ROOT, file_name),
        os.path.join(REPO_ROOT, "datasets", file_name),
        os.path.join(SCRIPT_DIR, file_name),  # script ile aynı klasör
        os.path.join(REPO_ROOT, "Labs-Umut", "lab4", file_name),  # repo içi lab4 konumu
        os.path.join(".", file_name),  # mevcut klasör
        os.path.join(FALLBACK_ROOT, file_name),
    ]
    last_err = None
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                print(f"[OK] Veri yüklendi: {p}")
                return df
            except Exception as e:
                last_err = e
                print(f"[HATA] Okuma başarısız: {p} -> {e}")
    raise FileNotFoundError(f"'{file_name}' bulunamadı veya okunamadı. Son hata: {last_err}")


# ==== YARDIMCI: NUMERİK DÖNÜŞÜM ===============================================
def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Verilen sütunu güvenli biçimde numerik seriye çevirip NaN'ları atar."""
    return pd.to_numeric(df[col], errors="coerce").dropna()


# ==== TANIMLAYICI İSTATİSTİKLER ==============================================
def calculate_descriptive_stats(data: pd.DataFrame, column: str) -> Dict[str, float]:
    """
    Belirtilen sayısal sütun için tanımlayıcı istatistikleri döndürür.
    - mean, median, mode, std, var, range, iqr, skew, kurtosis, q1, q2, q3, min, max
    """
    col = _find_column(
        [column, "strength_mpa", "compressive_strength", "strength", "fck", "fc", "value"],
        data.columns,
    )
    if col is None:
        raise KeyError("İstenen değer sütunu bulunamadı.")
    s = _to_numeric_series(data, col)
    if s.empty:
        raise ValueError(f"'{col}' sütunu numerik değil ya da boş.")

    q1, q2, q3 = np.percentile(s, [25, 50, 75])
    mode_vals = s.mode()
    mode_val = float(mode_vals.iloc[0]) if not mode_vals.empty else float("nan")
    stats_dict = {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "mode": mode_val,
        "std": float(s.std(ddof=1)),
        "var": float(s.var(ddof=1)),
        "range": float(s.max() - s.min()),
        "iqr": float(q3 - q1),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurtosis()),
        "q1": float(q1),
        "q2_median": float(q2),
        "q3": float(q3),
        "min": float(s.min()),
        "max": float(s.max()),
        "_column_used": col,
        "_count": int(len(s)),
    }
    return stats_dict


# ==== GÖRSEL: DAĞILIM + İŞARETLER =============================================
def plot_distribution(
    data: pd.DataFrame,
    column: str,
    title: str,
    save_path: Optional[str] = None,
) -> None:
    """Histogram+KDE; mean/median/mode çizgileri; ±1–3σ bantları."""
    col = _find_column([column, "strength_mpa", "compressive_strength", "strength", "value"], data.columns)
    if col is None:
        raise KeyError("Dağılım grafiği için sütun bulunamadı.")
    s = _to_numeric_series(data, col)
    st = calculate_descriptive_stats(data, col)

    ax = sns.histplot(s, bins="auto", kde=True, stat="density", alpha=0.35, edgecolor="k")
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")

    mean_v, med_v, mod_v, std_v = st["mean"], st["median"], st["mode"], st["std"]

    plt.axvline(mean_v, linestyle="--", linewidth=2, label=f"Mean={mean_v:.2f}")
    plt.axvline(med_v, linestyle="-.", linewidth=2, label=f"Median={med_v:.2f}")
    plt.axvline(mod_v, linestyle=":", linewidth=2, label=f"Mode={mod_v:.2f}")

    # ±σ bantları (hafif şeffaf): yalnızca 1σ'i etiketle, diğerleri gölge kalsın
    for k in [1, 2, 3]:
        plt.axvspan(mean_v - k * std_v, mean_v + k * std_v, alpha=0.06, label="±1σ" if k == 1 else None)

    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ==== GÖRSEL: BOXPlot =========================================================
def plot_boxplot_single(
    data: pd.DataFrame,
    column: str,
    title: str,
    save_path: Optional[str] = None,
) -> None:
    """Tek bir sayısal sütun için boxplot çizer."""
    col = _find_column([column, "strength_mpa", "compressive_strength", "strength", "value"], data.columns)
    if col is None:
        raise KeyError("Boxplot için sütun bulunamadı.")
    s = _to_numeric_series(data, col)
    sns.boxplot(y=s)
    plt.title(title)
    plt.ylabel(col)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ==== DAĞILIM UYDURMA (NORMAL) + OVERLAY ======================================
def fit_distribution(
    data: pd.DataFrame,
    column: str,
    distribution_type: str = "normal",
) -> Dict[str, float]:
    """Şimdilik Normal fit (μ, σ) döndürür."""
    col = _find_column([column, "strength_mpa", "compressive_strength", "strength", "value"], data.columns)
    if col is None:
        raise KeyError("Fit için sütun bulunamadı.")
    s = _to_numeric_series(data, col)

    if distribution_type.lower() == "normal":
        mu, sigma = norm.fit(s)
        return {"distribution": "normal", "mean": float(mu), "std": float(sigma), "_column_used": col}
    else:
        raise NotImplementedError("Şimdilik yalnızca 'normal' fit desteklenir.")


def plot_distribution_fitting(
    data: pd.DataFrame,
    column: str,
    fitted_dist: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Histogram + fitted normal PDF + sentetik KDE overlay."""
    col = _find_column([column, "strength_mpa", "compressive_strength", "strength", "value"], data.columns)
    if col is None:
        raise KeyError("Fit grafiği için sütun bulunamadı.")
    s = _to_numeric_series(data, col)

    if fitted_dist is None:
        fitted_dist = fit_distribution(data, column=col)

    mu, sigma = fitted_dist["mean"], fitted_dist["std"]

    ax = sns.histplot(s, bins="auto", stat="density", alpha=0.35, edgecolor="k", label="Real Data")
    x_grid = np.linspace(s.min(), s.max(), 300)
    plt.plot(x_grid, norm.pdf(x_grid, mu, sigma), linewidth=2, label=f"Normal PDF (μ={mu:.2f}, σ={sigma:.2f})")

    synthetic = np.random.normal(mu, sigma, size=max(500, 5 * len(s)))
    sns.kdeplot(synthetic, linewidth=1.6, label="Synthetic KDE")

    plt.title("Distribution Fitting: Real vs Normal Fit")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ==== QQ-PLOT (NORMAL) ========================================================
def plot_qq_normal(
    data: pd.DataFrame,
    column: str,
    save_path: Optional[str] = None,
) -> None:
    """Normalite için QQ-plot üretir."""
    col = _find_column([column, "strength_mpa", "compressive_strength", "strength", "value"], data.columns)
    if col is None:
        raise KeyError("QQ-plot için sütun bulunamadı.")
    s = _to_numeric_series(data, col)

    stats.probplot(s, dist="norm", plot=plt)
    plt.title("QQ-Plot (Normal)")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ==== MALZEME KARŞILAŞTIRMA (BOXPLOT) ========================================
def plot_material_comparison(
    data: pd.DataFrame,
    value_candidates: Optional[List[str]] = None,
    group_candidates: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Grup sütununa göre değer dağılımlarını kutu grafikle karşılaştırır."""
    value_candidates = value_candidates or [
        "strength_mpa", "compressive_strength", "yield_strength", "strength", "value",
        "ultimate_strength", "tensile_strength",
    ]
    group_candidates = group_candidates or [
        "material_type", "material", "type", "category", "class", "group",
    ]

    val_col = _find_column(value_candidates, data.columns)
    grp_col = _find_column(group_candidates, data.columns)

    if val_col is None or grp_col is None:
        # Yükseltmek yerine uyarı verip devam edelim; rapor üretilsin
        print("[UYARI] Material comparison atlandı (sütun bulunamadı).")
        print("Mevcut sütunlar:", list(data.columns))
        return

    df = data[[val_col, grp_col]].copy()
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna()

    if df.empty:
        print("[UYARI] Material comparison: uygun veri yok (hepsi NaN?).")
        return

    sns.boxplot(x=grp_col, y=val_col, data=df)
    plt.title("Material Strength Comparison (Boxplots)")
    plt.xlabel(grp_col)
    plt.ylabel(val_col)
    plt.xticks(rotation=15)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ==== STRUCTURAL LOADS – DAĞILIM GRAFİĞİ ======================================
def plot_structural_loads_distribution(
    data: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Yük verisinin dağılımını (hist+KDE) çizer; sütun ismini esnek arar."""
    load_col = _find_column(
        ["load_kn", "load", "loads", "applied_load", "service_load", "factored_load", "kN", "kn"],
        data.columns,
    )
    if load_col is None:
        print("[UYARI] Structural loads grafiği atlandı (yük sütunu bulunamadı).")
        print("Mevcut sütunlar:", list(data.columns))
        return

    s = _to_numeric_series(data, load_col)
    if s.empty:
        print("[UYARI] Structural loads: uygun sayısal veri yok.")
        return

    sns.histplot(s, bins="auto", kde=True, stat="density", alpha=0.35, edgecolor="k")
    plt.title("Structural Loads – Distribution")
    plt.xlabel(load_col)
    plt.ylabel("Density")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ==== OLASILIK HESAPLARI ======================================================
def prob_bernoulli_mean_var(p: float) -> Tuple[float, float]:
    """Bernoulli(p) için (mean, var)."""
    return p, p * (1 - p)


def prob_binomial_pmf(n: int, p: float, k: int) -> float:
    """Binom(n,p): P(X=k)."""
    return float(binom.pmf(k, n, p))


def prob_poisson_pmf(lmbda: float, k: int) -> float:
    """Poisson(λ): P(X=k)."""
    return float(poisson.pmf(k, lmbda))


def prob_uniform_mean_var(a: float, b: float) -> Tuple[float, float]:
    """Uniform[a,b]: (mean, var)."""
    return (a + b) / 2.0, ((b - a) ** 2) / 12.0


def prob_normal_interval(mean: float, std: float, x_lower: Optional[float], x_upper: Optional[float]) -> float:
    """Normal(μ,σ): P(x_lower < X < x_upper) veya tek taraflı olasılık."""
    if x_lower is not None and x_upper is not None:
        return float(norm.cdf(x_upper, mean, std) - norm.cdf(x_lower, mean, std))
    elif x_lower is not None:
        return float(1.0 - norm.cdf(x_lower, mean, std))
    elif x_upper is not None:
        return float(norm.cdf(x_upper, mean, std))
    else:
        raise ValueError("x_lower ve/veya x_upper belirtilmeli.")


def prob_exponential_cdf(mean: float, x: float) -> float:
    """Üstel(mean): P(X <= x)."""
    lam = 1.0 / mean
    return float(1 - np.exp(-lam * x))


# ==== BAYES ===================================================================
def apply_bayes_theorem(prior: float, sensitivity: float, specificity: float) -> float:
    """
    Bayes: Posterior P(D|+) = (Se * P(D)) / (Se*P(D) + (1-Sp)*(1-P(D)))
    prior: P(D)         — hasar var olasılığı (base rate)
    sensitivity: Se     — testin hasarı yakalama oranı
    specificity: Sp     — testin sağlamı doğru tanıma oranı
    """
    true_pos = sensitivity * prior
    false_pos = (1 - specificity) * (1 - prior)
    return float(true_pos / (true_pos + false_pos))


def plot_probability_tree_bayes(
    prior: float, sensitivity: float, specificity: float, save_path: Optional[str] = None
) -> None:
    """Basit bir olasılık ağacı (D/~D → +/−) çizer."""
    p_d = prior
    p_nd = 1 - prior
    p_pos_d = sensitivity
    p_neg_d = 1 - sensitivity
    p_neg_nd = specificity
    p_pos_nd = 1 - specificity

    root = (0.05, 0.5)
    d_node = (0.35, 0.7)
    nd_node = (0.35, 0.3)
    d_pos = (0.75, 0.85)
    d_neg = (0.75, 0.55)
    nd_pos = (0.75, 0.45)
    nd_neg = (0.75, 0.15)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    ax.text(*root, "Start", fontsize=11, ha="center", va="center", bbox=dict(boxstyle="round", fc="white"))
    ax.plot([root[0], d_node[0]], [root[1], d_node[1]], lw=2)
    ax.plot([root[0], nd_node[0]], [root[1], nd_node[1]], lw=2)
    ax.text((root[0]+d_node[0])/2, (root[1]+d_node[1])/2, f"P(D)={p_d:.2f}", fontsize=10)
    ax.text((root[0]+nd_node[0])/2, (root[1]+nd_node[1])/2, f"P(~D)={p_nd:.2f}", fontsize=10)

    ax.text(*d_node, "Damage (D)", fontsize=11, ha="center", va="center", bbox=dict(boxstyle="round", fc="white"))
    ax.text(*nd_node, "No Damage (~D)", fontsize=11, ha="center", va="center", bbox=dict(boxstyle="round", fc="white"))

    ax.plot([d_node[0], d_pos[0]], [d_node[1], d_pos[1]], lw=2)
    ax.plot([d_node[0], d_neg[0]], [d_node[1], d_neg[1]], lw=2)
    ax.text((d_node[0]+d_pos[0])/2, (d_node[1]+d_pos[1])/2, f"P(+|D)={p_pos_d:.2f}", fontsize=10)
    ax.text((d_node[0]+d_neg[0])/2, (d_node[1]+d_neg[1])/2, f"P(-|D)={p_neg_d:.2f}", fontsize=10)

    ax.plot([nd_node[0], nd_pos[0]], [nd_node[1], nd_pos[1]], lw=2)
    ax.plot([nd_node[0], nd_neg[0]], [nd_node[1], nd_neg[1]], lw=2)
    ax.text((nd_node[0]+nd_pos[0])/2, (nd_node[1]+nd_pos[1])/2, f"P(+|~D)={p_pos_nd:.2f}", fontsize=10)
    ax.text((nd_node[0]+nd_neg[0])/2, (nd_node[1]+nd_neg[1])/2, f"P(-|~D)={p_neg_nd:.2f}", fontsize=10)

    ax.text(*d_pos, "D & +", fontsize=10, ha="center", va="center")
    ax.text(*d_neg, "D & -", fontsize=10, ha="center", va="center")
    ax.text(*nd_pos, "~D & +", fontsize=10, ha="center", va="center")
    ax.text(*nd_neg, "~D & -", fontsize=10, ha="center", va="center")

    ax.set_title("Bayes Probability Tree: Structural Damage Test", fontsize=12)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ==== RAPOR OLUŞTURMA =========================================================
def create_statistical_report(
    stats_concrete: Dict[str, float],
    fit_info: Dict[str, float],
    probs: Dict[str, float],
    material_summary: str,
    output_file: str,
) -> None:
    """Özet raporu .txt olarak yazar."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    lines = []
    lines.append("Lab 4 Statistical Report")
    lines.append("=" * 72)
    lines.append("")

    # Beton istatistikleri
    lines.append("Concrete Strength – Descriptive Statistics")
    for k, v in stats_concrete.items():
        if not k.startswith("_"):
            lines.append(f"- {k}: {v}")
    lines.append("")

    # Fit bilgisi
    lines.append("Fitted Distribution (Concrete Strength)")
    lines.append(f"- Type: {fit_info.get('distribution', 'normal')}")
    lines.append(f"- Mean (mu): {fit_info.get('mean'):.4f}")
    lines.append(f"- Std (sigma): {fit_info.get('std'):.4f}")
    lines.append("")

    # Olasılıklar
    lines.append("Probability Scenarios")
    for k, v in probs.items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")

    # Malzeme özeti
    lines.append("Material Comparison (Summary)")
    lines.append(textwrap.fill(material_summary or "N/A", width=72))
    lines.append("")

    # Kısa çıkarımlar
    lines.append("Engineering Notes")
    bullets = [
        "Yüksek std/iqr kalite kontrol eşiklerini gözden gerektirir.",
        "Mean≈Median ve düşük skew normalite varsayımı için olumlu bir işarettir.",
        "Bayes ile düşük base-rate durumlarında yalancı-pozitif etkisi sayısallaşır.",
        "Gruplar arası fark anlamlı ise proses varyasyon kaynaklarına bakılmalı.",
    ]
    for b in bullets:
        lines.append(f"- {b}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Rapor yazıldı: {output_file}")


# ==== ANA AKIŞ ================================================================
def main():
    # Çıktı klasörü
    out_dir = os.path.join("Labs-Umut", "lab4")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Verileri yükle
    df_concrete = load_data("concrete_strength.csv")
    df_loads = load_data("structural_loads.csv")
    df_material = load_data("material_properties.csv")

    # 2) Eksik değerleri kaba doldurma (md gereği bir temizlik turu)
    def _fill_basic(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
                df[c] = df[c].fillna(df[c].median())
            else:
                if df[c].isna().any():
                    mode_vals = df[c].mode()
                    df[c] = df[c].fillna(mode_vals.iloc[0] if not mode_vals.empty else "Unknown")
        return df

    df_concrete = _fill_basic(df_concrete)
    df_loads = _fill_basic(df_loads)
    df_material = _fill_basic(df_material)

    # 3) Beton istatistikleri
    stats_conc = calculate_descriptive_stats(df_concrete, column="strength_mpa")
    print("\n[RESULT] Concrete descriptive stats")
    for k, v in stats_conc.items():
        print(f"{k}: {v}")

    # 4) Görseller – Concrete
    plot_distribution(
        df_concrete,
        column=stats_conc["_column_used"],
        title="Concrete Strength – Distribution",
        save_path=os.path.join(out_dir, "concrete_strength_distribution.png"),
    )
    plot_boxplot_single(
        df_concrete,
        column=stats_conc["_column_used"],
        title="Concrete Strength – Boxplot",
        save_path=os.path.join(out_dir, "concrete_strength_boxplot.png"),
    )

    # 5) Fit + QQ-plot
    fit_info = fit_distribution(df_concrete, column=stats_conc["_column_used"])
    print(f"\n[FIT] Normal -> mu={fit_info['mean']:.4f}, sigma={fit_info['std']:.4f}")

    plot_distribution_fitting(
        df_concrete,
        column=fit_info["_column_used"],
        fitted_dist=fit_info,
        save_path=os.path.join(out_dir, "distribution_fitting.png"),
    )
    plot_qq_normal(
        df_concrete,
        column=fit_info["_column_used"],
        save_path=os.path.join(out_dir, "qqplot_concrete.png"),
    )

    # 6) Loads dağılımı (varsa)
    plot_structural_loads_distribution(
        df_loads,
        save_path=os.path.join(out_dir, "structural_loads_distribution.png"),
    )

    # 7) Malzeme karşılaştırması
    plot_material_comparison(
        df_material,
        save_path=os.path.join(out_dir, "material_comparison_boxplot.png"),
    )

    # Malzeme istatistik özeti (grup ortalama-std-count tablosu yazıya dök)
    mat_val = _find_column(
        ["strength_mpa", "compressive_strength", "yield_strength", "strength", "value"],
        df_material.columns,
    )
    mat_grp = _find_column(["material_type", "material", "type", "category", "class", "group"], df_material.columns)
    material_summary = "N/A"
    if mat_val and mat_grp:
        dfm = df_material[[mat_val, mat_grp]].copy()
        dfm[mat_val] = pd.to_numeric(dfm[mat_val], errors="coerce")
        g = dfm.dropna().groupby(mat_grp)[mat_val].agg(["mean", "std", "count"]).sort_values("mean")
        material_summary = g.to_string()
        print("\n[Material Comparison] mean/std/count by group")
        print(g)
    else:
        print("\n[Material Comparison] Özet atlandı (sütun eşleşmedi).")

    # 8) Olasılık senaryoları (örnekler)
    p_bern_m, p_bern_v = prob_bernoulli_mean_var(0.97)
    p_bin3 = prob_binomial_pmf(100, 0.05, 3)
    p_bin_le5 = float(binom.cdf(5, 100, 0.05))
    p_pois8 = prob_poisson_pmf(10.0, 8)
    p_pois_gt15 = float(1 - poisson.cdf(15, 10.0))
    u_mean, u_var = prob_uniform_mean_var(0.0, 100.0)
    p_norm_gt280 = prob_normal_interval(250.0, 15.0, x_lower=280.0, x_upper=None)
    p95 = float(norm.ppf(0.95, 250.0, 15.0))
    p_exp_lt500 = prob_exponential_cdf(1000.0, 500.0)
    p_exp_gt1500 = 1.0 - prob_exponential_cdf(1000.0, 1500.0)

    posterior = apply_bayes_theorem(prior=0.05, sensitivity=0.95, specificity=0.90)
    print(f"\n[Bayes] P(Damage | Positive) = {posterior:.6f}")

    plot_probability_tree_bayes(
        prior=0.05, sensitivity=0.95, specificity=0.90,
        save_path=os.path.join(out_dir, "bayes_probability_tree.png"),
    )

    # 9) Mini dashboard (4 panel)
    try:
        col_strength = stats_conc["_column_used"]
        series = _to_numeric_series(df_concrete, col_strength)
        x_grid = np.linspace(series.min(), series.max(), 200)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Statistical Summary Dashboard", fontsize=14)

        sns.histplot(series, bins="auto", kde=True, stat="density", ax=axes[0, 0], alpha=0.35, edgecolor="k")
        axes[0, 0].set_title("Concrete Strength – Hist/KDE")

        sns.boxplot(y=series, ax=axes[0, 1])
        axes[0, 1].set_title("Concrete – Boxplot")

        axes[1, 0].plot(x_grid, norm.pdf(x_grid, fit_info["mean"], fit_info["std"]), lw=2)
        axes[1, 0].set_title("Fitted Normal PDF")

        axes[1, 1].plot(x_grid, norm.cdf(x_grid, fit_info["mean"], fit_info["std"]), lw=2)
        axes[1, 1].set_title("Normal CDF (μ, σ)")

        for ax in axes.flat:
            ax.set_xlabel("")
            ax.set_ylabel("")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(out_dir, "statistical_summary_dashboard.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("[UYARI] Dashboard oluşturulamadı:", e)

    # 10) Rapor yaz
    probs = {
        "Bernoulli_mean": p_bern_m,
        "Bernoulli_var": p_bern_v,
        "Binomial_P(X=3)": p_bin3,
        "Binomial_P(X<=5)": p_bin_le5,
        "Poisson_P(X=8)": p_pois8,
        "Poisson_P(X>15)": p_pois_gt15,
        "Uniform_mean_0_100": u_mean,
        "Uniform_var_0_100": u_var,
        "Normal_P(X>280)_mu250_sigma15": p_norm_gt280,
        "Normal_95th_percentile_mu250_sigma15": p95,
        "Exponential_P(fail<=500h)_mean1000": p_exp_lt500,
        "Exponential_P(survive>1500h)_mean1000": p_exp_gt1500,
        "Bayes_P(D|+)": posterior,
    }

    create_statistical_report(
        stats_concrete=stats_conc,
        fit_info=fit_info,
        probs=probs,
        material_summary=material_summary,
        output_file=os.path.join(out_dir, "lab4_statistical_report.txt"),
    )

    # 11) Konsol özeti
    print("\n[SUMMARY] Üretilen dosyalar:")
    for fn in [
        "concrete_strength_distribution.png",
        "concrete_strength_boxplot.png",
        "distribution_fitting.png",
        "qqplot_concrete.png",
        "structural_loads_distribution.png",
        "material_comparison_boxplot.png",
        "bayes_probability_tree.png",
        "statistical_summary_dashboard.png",
        "lab4_statistical_report.txt",
    ]:
        print(" -", os.path.join(out_dir, fn))


# ==== ÇALIŞTIRICI =============================================================
if __name__ == "__main__":
    main()
