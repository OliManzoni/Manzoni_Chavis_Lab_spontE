import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize, integrate, ndimage
import tempfile
import os
import pandas as pd

# Tentative d'import de pyabf
try:
    import pyabf
except ImportError:
    st.error("Le module pyabf n'est pas installé. Exécutez : pip install pyabf")

# --- CONFIGURATION ---
st.set_page_config(page_title="sEPSC Pipeline", layout="wide")

if 'fs_nyquist' not in st.session_state:
    st.session_state.fs_nyquist = 5000.0

# --- LANGUAGE SELECTION ---
lang = st.sidebar.selectbox("Language / Langue", ["English", "Français"])

# Textes explicatifs (Markdown + LaTeX)
THEORY_EN = """
### 🔬 Biophysical and Mathematical Principles

#### 1. Dynamic Detrending & Bessel Filter (Preprocessing)
**Dynamic Detrending:** Slow fluctuations in the holding current can shift the local baseline and bias area measurements. The algorithm applies a **sliding median filter** (e.g., 500 ms window) to track and subtract these slow drifts without distorting fast synaptic transients, locking the global baseline strictly to 0 pA.
**Bessel Filter:** Unlike Butterworth filters which introduce "ringing" on fast transients, the **Bessel filter** has a strictly linear phase response. All frequencies are delayed equally, preserving the exact waveform and **Rise Time** of the sEPSC.

#### 2. Multi-Scale Template Matching
Synaptic events originating from distal dendrites undergo **passive dendritic filtering**, resulting in slower, broader waveforms. 
Instead of a single template, the algorithm convolves the trace with multiple bi-exponential templates having varying decay constants ($\tau = 2, 5, 10, 15$ ms). A sliding cross-correlation maximizes the detection of heterogeneous events buried in high RMS noise.

#### 3. Kinetics: Rise Time 10-90%
The 10-90% rise time is calculated to avoid the noise present at the absolute peak and the baseline drift at the onset. The algorithm uses **linear interpolation** between the digitized sampling points to pinpoint the exact millisecond the current crosses these thresholds, bypassing the limits of the sampling frequency.

#### 4. Decay Estimation via Charge Integration
Performing non-linear curve fitting on hundreds of noisy spontaneous events is computationally heavy. Assuming a simple exponential decay for an AMPA current ($I(t) = I_{max} e^{-t/\\tau}$), the total charge (Area) is the integral:
$$ \\text{Area} = \\int_{0}^{\\infty} I_{max} e^{-t/\\tau} dt = I_{max} \\cdot \\tau $$
Therefore, the decay constant $\\tau$ is rapidly and robustly estimated:
$$ \\tau \\approx \\frac{\\text{Area}}{\\text{Amplitude}} $$
"""

THEORY_FR = """
### 🔬 Principes Biophysiques et Mathématiques

#### 1. Detrending Dynamique & Filtre de Bessel (Prétraitement)
**Detrending Dynamique :** Les fluctuations lentes du courant de maintien peuvent fausser la ligne de base. L'algorithme applique un **filtre médian glissant** (ex: fenêtre de 500 ms) pour suivre et soustraire ces dérives sans affecter les transitoires rapides, verrouillant la ligne de base strictement à 0 pA.
**Filtre de Bessel :** Contrairement au Butterworth qui introduit des oscillations ("ringing"), le **filtre de Bessel** possède une réponse en phase linéaire. Il préserve la forme d'onde exacte du sEPSC, une condition absolue pour mesurer le **Rise Time**.

#### 2. Détection par "Template Matching" Multi-Échelle
Les événements dendritiques subissent un **filtrage passif** qui ralentit leur forme d'onde. Au lieu d'un seul modèle, l'algorithme génère des modèles bi-exponentiels avec différentes constantes de temps de décroissance ($\tau = 2, 5, 10, 15$ ms). Une corrélation croisée glissante identifie ces événements hétérogènes.

#### 3. Cinétique : Rise Time 10-90%
Le temps de montée 10-90% s'affranchit du bruit au pic. L'algorithme utilise une **interpolation linéaire** entre les points d'échantillonnage pour trouver la milliseconde exacte où le signal franchit ces seuils.

#### 4. Estimation du Decay par Intégration de la Charge
Faire un *curve-fitting* sur des événements bruités échoue souvent. En supposant une décroissance exponentielle simple ($I(t) = I_{max} e^{-t/\\tau}$), la charge totale (Aire) est :
$$ \\text{Aire} = \\int_{0}^{\\infty} I_{max} e^{-t/\\tau} dt = I_{max} \\cdot \\tau $$
La constante de temps $\\tau$ peut donc être estimée robustement :
$$ \\tau \\approx \\frac{\\text{Aire}}{\\text{Amplitude}} $$
"""

T = {
    "English": {
        "title": "# sEPSC Expert Pipeline: Preprocessing, Kinetics & Export",
        "branding": "Chavis Lab - Biophysics",
        "readme_link": "📖 View README (Documentation)",
        "cite_header": "🎓 Cite this App",
        "cite_text": "If you use this tool, please cite:",
        "tab_analysis": "📈 Analysis Pipeline",
        "tab_theory": "📚 Biophysics & Math Theory",
        "theory_text": THEORY_EN,
        "sb_preproc": "1. Preprocessing (Baseline & Denoising)",
        "baseline_method": "Baseline Correction Mode",
        "dyn_detrend": "Dynamic Detrending (Rolling Median)",
        "stat_detrend": "Static Global Median",
        "cutoff": "Bessel Cutoff (Hz)",
        "nyquist_warn": "⚠️ Limited by Nyquist frequency",
        "sb_detec": "2. Multi-Scale Detection",
        "threshold": "Z-Score Threshold",
        "sb_kinetics": "3. Kinetics & Filters",
        "decay_thresh": "Decay Threshold (ms)",
        "rise_thresh": "Rise Time Threshold (ms)",
        "calc_raw": "Calculate on RAW trace",
        "amp_filter": "Amplitude Filter (>7pA)",
        "sb_viz": "4. Visualization",
        "zoom_y": "Zoom Y (pA)",
        "zoom_x": "Zoom X Window",
        "x_start": "Start (s)",
        "x_end": "End (s)",
        "uploader": "Upload .abf",
        "viz_header": "Visualization & Detection",
        "export_header": "📥 Export Results",
        "btn_events": "📁 Download Individual Events",
        "btn_summary": "📊 Download Population Analysis",
        "col_time": "Time (s)",
        "col_amp": "Amplitude (pA)",
        "col_rise": "Rise Time 10-90% (ms)",
        "col_decay": "Estimated Decay (ms)",
        "col_area": "Area (pA.ms)",
        "col_iei": "IEI (ms)"
    },
    "Français": {
        "title": "# Pipeline Expert sEPSC : Prétraitement, Cinétique & Exportation",
        "branding": "Chavis Lab - Biophysique",
        "readme_link": "📖 Voir le README (Documentation)",
        "cite_header": "🎓 Citer cette App",
        "cite_text": "Si vous utilisez cet outil, merci de citer :",
        "tab_analysis": "📈 Pipeline d'Analyse",
        "tab_theory": "📚 Théorie Biophysique & Maths",
        "theory_text": THEORY_FR,
        "sb_preproc": "1. Prétraitement (Baseline & Denoising)",
        "baseline_method": "Mode de correction de Ligne de Base",
        "dyn_detrend": "Detrending Dynamique (Médiane Glissante)",
        "stat_detrend": "Médiane Globale Statique",
        "cutoff": "Coupure Bessel (Hz)",
        "nyquist_warn": "⚠️ Limité par la fréquence de Nyquist",
        "sb_detec": "2. Détection Multi-Scale",
        "threshold": "Seuil Z-Score",
        "sb_kinetics": "3. Cinétique & Filtres",
        "decay_thresh": "Seuil maximal Decay (ms)",
        "rise_thresh": "Seuil maximal Rise Time (ms)",
        "calc_raw": "Calculer sur trace BRUTE",
        "amp_filter": "Filtre Amplitude (>7pA)",
        "sb_viz": "4. Visualisation",
        "zoom_y": "Zoom Y (pA)",
        "zoom_x": "Fenêtre Zoom X",
        "x_start": "Début (s)",
        "x_end": "Fin (s)",
        "uploader": "Charger .abf",
        "viz_header": "Visualisation & Détection",
        "export_header": "📥 Exportation des Résultats",
        "btn_events": "📁 Télécharger Événements Individuels",
        "btn_summary": "📊 Télécharger Analyse Population",
        "col_time": "Temps (s)",
        "col_amp": "Amplitude (pA)",
        "col_rise": "Rise Time 10-90% (ms)",
        "col_decay": "Decay Estimé (ms)",
        "col_area": "Aire (pA.ms)",
        "col_iei": "IEI (ms)"
    }
}[lang]

# --- LIEN VERS LE README & DOI ---
st.sidebar.markdown(f"**[{T['readme_link']}](https://github.com/OliManzoni/Manzoni_Chavis_Lab_spontE/blob/main/README.md)**")
st.sidebar.markdown(f"### {T['cite_header']}")
st.sidebar.markdown("[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19915015.svg)](https://doi.org/10.5281/zenodo.19915015)")
st.sidebar.caption(f"{T['cite_text']}\n*Manzoni, O. J. (2026). Manzoni_Chavis_Lab_spontE. Zenodo. https://doi.org/10.5281/zenodo.19915015*")
st.sidebar.divider()

# --- EN-TÊTE INSTITUTIONNEL ---
col_l, col_r = st.columns([2, 5]) 
with col_l:
    try: st.image("logo_chavis_final.png", width=360) 
    except: st.info(T["branding"]) 
with col_r:
    st.markdown(T["title"])

st.divider()

# --- CRÉATION DES ONGLETS ---
tab_analysis, tab_theory = st.tabs([T["tab_analysis"], T["tab_theory"]])

# ==========================================
# ONGLET 2 : THEORIE BIOPHYSIQUE
# ==========================================
with tab_theory:
    st.markdown(T["theory_text"])

# ==========================================
# ONGLET 1 : PIPELINE D'ANALYSE
# ==========================================
with tab_analysis:
    # --- FONCTIONS MATHÉMATIQUES ---
    def apply_dynamic_detrending(data, fs, window_ms=500):
        """Soustrait la dérive lente en utilisant un filtre médian glissant."""
        kernel_size = int((window_ms / 1000.0) * fs)
        if kernel_size % 2 == 0: kernel_size += 1 # Le kernel doit être impair
        baseline = ndimage.median_filter(data, size=kernel_size)
        return data - baseline

    def apply_bessel_filter(data, fs, cutoff=2000, order=4):
        nyquist = 0.5 * fs
        effective_cutoff = min(cutoff, nyquist * 0.95)
        normal_cutoff = effective_cutoff / nyquist
        b, a = signal.bessel(order, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, data)

    def calculate_rise_time_expert(segment_y, dt):
        try:
            peak_idx = np.argmax(segment_y)
            if peak_idx < 3: return 0
            rising_limb = segment_y[:peak_idx + 1]
            t_vec = np.arange(len(rising_limb)) * dt
            peak_val = rising_limb[-1]
            y10, y90 = 0.10 * peak_val, 0.90 * peak_val
            t10 = np.interp(y10, rising_limb, t_vec)
            t90 = np.interp(y90, rising_limb, t_vec)
            return t90 - t10
        except: return 0

    # --- SIDEBAR & FILTRES ---
    st.sidebar.header(T["sb_preproc"])
    baseline_mode = st.sidebar.radio(T["baseline_method"], [T["dyn_detrend"], T["stat_detrend"]], index=0)
    use_bessel = st.sidebar.checkbox("Bessel Filter", value=True)
    cutoff = st.sidebar.slider(T["cutoff"], 100, int(st.session_state.fs_nyquist), 2000)

    st.sidebar.header(T["sb_detec"])
    threshold = st.sidebar.slider(T["threshold"], 1.0, 8.0, 2.5)

    st.sidebar.header(T["sb_kinetics"])
    use_decay_filter = st.sidebar.checkbox("Filter Decay", value=True)
    decay_limit = st.sidebar.number_input(T["decay_thresh"], value=3.0, step=0.5)

    use_rise_filter = st.sidebar.checkbox("Filter Rise Time", value=True)
    rise_limit = st.sidebar.number_input(T["rise_thresh"], value=0.5, step=0.1)

    use_amp_filter = st.sidebar.checkbox(T["amp_filter"], value=True)
    calc_on_raw = st.sidebar.checkbox(T["calc_raw"], value=False)

    st.sidebar.header(T["sb_viz"])
    y_zoom = st.sidebar.slider(T["zoom_y"], -300, 100, (-80, 20))
    
    st.sidebar.markdown(f"**{T['zoom_x']}**")
    col_x1, col_x2 = st.sidebar.columns(2)
    x_start = col_x1.number_input(T["x_start"], value=10.0, step=0.5)
    x_end = col_x2.number_input(T["x_end"], value=11.0, step=0.5)
    x_zoom = (x_start, x_end)

    # --- LOGIQUE PRINCIPALE ---
    file = st.file_uploader(T["uploader"], type=["abf"])

    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.abf') as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        try:
            abf = pyabf.ABF(tmp_path)
            abf.setSweep(0)
            fs, times, dt = abf.dataRate, abf.sweepX, 1000/abf.dataRate
            
            nyquist_limit = fs / 2
            st.session_state.fs_nyquist = nyquist_limit
            if cutoff >= nyquist_limit:
                st.sidebar.warning(T["nyquist_warn"])

            # POINT CLÉ : Application de la correction de Ligne de Base
            if baseline_mode == T["dyn_detrend"]:
                with st.spinner("Application du Detrending Dynamique (Rolling Median)..."):
                    raw_data = apply_dynamic_detrending(abf.sweepY, fs, window_ms=500)
            else:
                raw_data = abf.sweepY - np.median(abf.sweepY)

            f_data = apply_bessel_filter(raw_data, fs, cutoff) if use_bessel else raw_data
            
            best_corr = np.zeros_like(f_data)
            default_decays = [2.0, 5.0, 10.0, 15.0]
            for d in default_decays:
                t_tmpl = np.arange(0, 20, dt)
                tmpl = (np.exp(-t_tmpl/d) - np.exp(-t_tmpl/0.5))
                tmpl /= np.max(np.abs(tmpl))
                best_corr = np.maximum(best_corr, signal.correlate(-f_data, tmpl, mode='same'))
            
            corr_z = (best_corr - np.mean(best_corr)) / np.std(best_corr)
            peaks, _ = signal.find_peaks(corr_z, height=threshold, distance=int(0.005 * fs))
            
            valid_ev = []
            k_trace = raw_data if calc_on_raw else f_data
            
            for i, p in enumerate(peaks):
                start, end = p - int(0.003*fs), p + int(0.015*fs)
                if start < 0 or end >= len(k_trace): continue
                
                # La soustraction locale reste pour peaufiner l'intégration
                l_base = np.mean(k_trace[p-int(0.005*fs):p-int(0.002*fs)])
                seg_inv = -(k_trace[start:end] - l_base)
                
                amp = np.max(seg_inv)
                rise_1090 = calculate_rise_time_expert(seg_inv, dt)
                area = integrate.trapezoid(seg_inv, dx=dt)
                
                estimated_decay = abs(area / amp) if amp > 0 else 0
                
                pass_amp = (not use_amp_filter or amp >= 7)
                pass_decay = (not use_decay_filter or estimated_decay <= decay_limit)
                pass_rise = (not use_rise_filter or rise_1090 <= rise_limit)
                
                if pass_amp and pass_decay and pass_rise:
                    ev = {'idx': p, 'time': times[p], 'amp': amp, 'rise': rise_1090, 'area': abs(area), 'decay': estimated_decay}
                    ev['iei'] = (times[p] - times[peaks[i-1]])*1000 if i>0 else np.nan
                    valid_ev.append(ev)

            # --- AFFICHAGE GRAPHIQUE ---
            st.subheader(T["viz_header"])
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[2,1]})
            ax1.plot(times, f_data, color='black', lw=0.4)
            if valid_ev: ax1.plot([e['time'] for e in valid_ev], [f_data[e['idx']] for e in valid_ev], 'o', color='#FF8C00', markersize=5)
            ax1.set_ylim(y_zoom)
            ax1.set_xlim(x_zoom)
            ax2.plot(times, corr_z, color='blue', alpha=0.5)
            ax2.axhline(threshold, color='red', ls='--')
            st.pyplot(fig1)

            # --- EXPORT & DISTRIBUTIONS ---
            if valid_ev:
                df = pd.DataFrame(valid_ev)
                st.divider()
                
                freq_hz = len(df) / times[-1]
                mean_iei_ms = df['iei'].mean()
                
                df_export = df[['time', 'amp', 'rise', 'decay', 'area', 'iei']].copy()
                df_export.rename(columns={'time': T['col_time'], 'amp': T['col_amp'], 'rise': T['col_rise'], 'decay': T['col_decay'], 'area': T['col_area'], 'iei': T['col_iei']}, inplace=True)
                
                st.subheader(T["export_header"])
                col_exp1, col_exp2 = st.columns(2)
                col_exp1.download_button(label=T["btn_events"], data=df_export.to_csv(index=False).encode('utf-8'), file_name='sEPSC_events.csv', mime='text/csv')
                
                # Export de la Population
                n_bins = 25
                counts_amp, bins_amp = np.histogram(df['amp'], bins=n_bins)
                counts_rise, bins_rise = np.histogram(df['rise'], bins=n_bins)
                iei_clean = df['iei'].dropna()
                counts_iei, bins_iei = np.histogram(iei_clean, bins=n_bins) if not iei_clean.empty else (np.zeros(n_bins), np.zeros(n_bins+1))

                metrics_keys = ["Total Events", "Frequency (Hz)", "Mean IEI (ms)", "Mean Amp (pA)", "Mean Rise (ms)", "Mean Area (pA.ms)"]
                metrics_vals = [len(df), freq_hz, mean_iei_ms, df['amp'].mean(), df['rise'].mean(), df['area'].mean()]
                pad_len = n_bins - len(metrics_keys)
                metrics_keys += [np.nan] * pad_len
                metrics_vals += [np.nan] * pad_len

                df_export_summary = pd.DataFrame({
                    'Metric': metrics_keys,
                    'Value': metrics_vals,
                    'Amp_Bin_Center_pA': (bins_amp[:-1] + bins_amp[1:]) / 2,
                    'Amp_Counts': counts_amp,
                    'Rise_Bin_Center_ms': (bins_rise[:-1] + bins_rise[1:]) / 2,
                    'Rise_Counts': counts_rise,
                    'IEI_Bin_Center_ms': (bins_iei[:-1] + bins_iei[1:]) / 2,
                    'IEI_Counts': counts_iei
                })

                col_exp2.download_button(label=T["btn_summary"], data=df_export_summary.to_csv(index=False).encode('utf-8'), file_name='sEPSC_population_analysis.csv', mime='text/csv')
                
                st.divider()
                st.subheader(f"Total Events: {len(valid_ev)} | Freq: {freq_hz:.2f} Hz")
                
                fig2, (ha, hb, hc) = plt.subplots(1, 3, figsize=(15, 4))
                ha.bar((bins_amp[:-1] + bins_amp[1:]) / 2, counts_amp, width=(bins_amp[1]-bins_amp[0])*0.9, color='gray')
                ha.set_title(T["col_amp"])
                
                hb.bar((bins_rise[:-1] + bins_rise[1:]) / 2, counts_rise, width=(bins_rise[1]-bins_rise[0])*0.9, color='#FF8C00')
                hb.set_title(T["col_rise"])
                
                if not iei_clean.empty:
                    hc.bar((bins_iei[:-1] + bins_iei[1:]) / 2, counts_iei, width=(bins_iei[1]-bins_iei[0])*0.9, color='salmon')
                    hc.set_title(T["col_iei"])
                    
                st.pyplot(fig2)

        except Exception as e: st.error(f"Error: {e}")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
