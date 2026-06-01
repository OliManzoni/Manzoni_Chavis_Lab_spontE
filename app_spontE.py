import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize, integrate, ndimage
import tempfile
import os
import pandas as pd

try:
    import pyabf
except ImportError:
    st.error("Le module pyabf n'est pas installé. Exécutez : pip install pyabf")

# --- CONFIGURATION ---
st.set_page_config(page_title="Manzoni Lab sEPSC Pipeline", layout="wide")

# Initialisation de la mémoire (Session State) pour la navigation temporelle
if 'fs_nyquist' not in st.session_state:
    st.session_state.fs_nyquist = 5000.0
if 'x_start' not in st.session_state:
    st.session_state.x_start = 10.0
if 'x_end' not in st.session_state:
    st.session_state.x_end = 11.0

# --- FONCTIONS DE NAVIGATION (Callbacks) ---
def scroll_left():
    window = st.session_state.x_end - st.session_state.x_start
    shift = window * 0.8
    new_start = max(0.0, st.session_state.x_start - shift)
    st.session_state.x_end = new_start + window
    st.session_state.x_start = new_start

def scroll_right():
    window = st.session_state.x_end - st.session_state.x_start
    shift = window * 0.8
    st.session_state.x_start += shift
    st.session_state.x_end += shift

# --- LANGUAGE SELECTION ---
lang = st.sidebar.selectbox("Language / Langue", ["English", "Français"])

T = {
    "English": {
        "title": "# Manzoni Lab: sEPSC Expert Pipeline",
        "sb_preproc": "1. Preprocessing (AMPA)",
        "baseline_method": "Baseline Mode",
        "dyn_detrend": "Dynamic Detrending (Median)",
        "stat_detrend": "Static Global Median",
        "cutoff": "Bessel Cutoff (Hz)",
        "sb_detec": "2. Detection Threshold",
        "threshold": "Z-Score Threshold",
        "sb_kinetics": "3. Kinetics Filters",
        "decay_thresh": "Max Decay (ms)",
        "rise_thresh": "Max Rise Time (ms)",
        "amp_filter": "Min Absolute Amplitude (pA)",
        "sb_viz": "4. Visualization & Navigation",
        "zoom_y": "Zoom Y (pA)",
        "x_start": "Start (s)",
        "x_end": "End (s)",
        "auto_z": "Auto-scale Z-score axis",
        "viz_header": "Visualization & Detection",
        "btn_left": "⬅️ Left",
        "btn_right": "Right ➡️"
    },
    "Français": {
        "title": "# Manzoni Lab : Pipeline Expert sEPSC",
        "sb_preproc": "1. Prétraitement (AMPA)",
        "baseline_method": "Mode de Ligne de Base",
        "dyn_detrend": "Detrending Dynamique (Médiane)",
        "stat_detrend": "Médiane Globale Statique",
        "cutoff": "Coupure Bessel (Hz)",
        "sb_detec": "2. Seuil de Détection",
        "threshold": "Seuil Z-Score",
        "sb_kinetics": "3. Filtres Cinétiques",
        "decay_thresh": "Decay Max (ms)",
        "rise_thresh": "Rise Time Max (ms)",
        "amp_filter": "Amplitude Absolue Min (pA)",
        "sb_viz": "4. Visualisation & Navigation",
        "zoom_y": "Zoom Y (pA)",
        "x_start": "Début (s)",
        "x_end": "Fin (s)",
        "auto_z": "Auto-ajustement axe Z",
        "viz_header": "Visualisation & Détection",
        "btn_left": "⬅️ Gauche",
        "btn_right": "Droite ➡️"
    }
}[lang]

st.title(T["title"])
st.divider()

# --- METHODOLOGIE & BIOPHYSIQUE (EXPANDER) ---
with st.expander("ℹ️ Méthodologie & Rigueur Biophysique / Biophysical Methodology", expanded=False):
    if lang == "Français":
        st.markdown("""
        ### 🔬 Résumé Pédagogique pour l'Analyse des sEPSC
        Ce pipeline est conçu pour extraire avec précision les courants postsynaptiques excitateurs spontanés (sEPSC), médiés principalement par les récepteurs **AMPA**. Voici la logique biophysique derrière l'algorithme :

        * **1. Prétraitement (Ligne de base & Bruit) :** * **Filtre Médian (Detrending) :** En patch-clamp, la membrane subit de lentes fluctuations. Contrairement à une simple moyenne, la médiane suit cette dérive de la ligne de base sans être artificiellement tirée vers le bas par les événements synaptiques eux-mêmes.
            * **Filtre de Bessel :** Un filtre passe-bas doux qui élimine le bruit électrique à haute fréquence tout en préservant la forme réelle (cinétique) des courants synaptiques.
        * **2. Détection par Gabarits (*Template Matching*) :** * L'algorithme ne se contente pas de chercher des pics dépassant un seuil fixe. Il glisse des modèles mathématiques (gabarits avec des constantes de temps $\\tau$ de 2 à 15 ms) sur la trace pour repérer la "signature" typique de l'ouverture et fermeture d'un canal AMPA.
            * **Z-Score :** Le signal de corrélation est normalisé. Un Z-Score de 2.5 signifie que l'événement se détache de 2.5 écarts-types au-dessus du bruit de fond, garantissant une détection objective, quelle que soit la qualité du *seal*.
        * **3. Extraction des Cinétiques :**
            * **Amplitude :** Reflète le nombre de récepteurs AMPA activés (réponse post-synaptique ou taille quantique). L'algorithme inverse mathématiquement les courants entrants (*inward*) pour faciliter la lecture.
            * **Rise Time (10-90%) :** Mesure le temps de montée. Un temps court (ex: 0.5 ms) indique une synapse proche du soma ; un temps long révèle un filtrage dendritique (synapse éloignée dont le courant s'est atténué en voyageant).
            * **Intervalle Inter-Événements (IEI) :** L'inverse de la fréquence. Reflète la probabilité de libération spontanée des vésicules par le neurone pré-synaptique.
        """)
    else:
        st.markdown("""
        ### 🔬 Pedagogical Summary for sEPSC Analysis
        This pipeline is designed to accurately extract spontaneous Excitatory Postsynaptic Currents (sEPSCs), primarily mediated by **AMPA** receptors. Here is the biophysical logic behind the algorithm:

        * **1. Preprocessing (Baseline & Noise):** * **Median Filter (Detrending):** Patch-clamp recordings often suffer from slow membrane fluctuations. Unlike a simple mean, a median filter tracks this baseline drift without being distorted by the fast synaptic events themselves.
            * **Bessel Filter:** A smooth low-pass filter that removes high-frequency electrical noise while preserving the true shape (kinetics) of the synaptic currents.
        * **2. Template Matching Detection:** * The algorithm does not simply look for peaks crossing a raw threshold. It slides mathematical models (templates with $\\tau$ decay constants ranging from 2 to 15 ms) across the trace to find the typical "signature" of AMPA channel gating.
            * **Z-Score:** The correlation signal is normalized. A Z-Score of 2.5 means the event stands 2.5 standard deviations above the background noise, ensuring objective detection regardless of the seal quality.
        * **3. Kinetics Extraction:**
            * **Amplitude:** Reflects the number of activated AMPA receptors (postsynaptic quantal size). The algorithm mathematically inverts the inward currents to facilitate data reading.
            * **Rise Time (10-90%):** Measures the rising phase. A short rise time (e.g., 0.5 ms) indicates a synapse close to the soma; a long rise time reveals dendritic filtering (a distant synapse whose current attenuated while traveling).
            * **Inter-Event Interval (IEI):** The inverse of frequency. It reflects the probability of spontaneous vesicle release from the presynaptic neuron.
        """)

# --- SIDEBAR ---
st.sidebar.header(T["sb_preproc"])
# L'option "Inward" est fixée mathématiquement pour les EPSCs, on allège l'interface
baseline_mode = st.sidebar.radio(T["baseline_method"], [T["dyn_detrend"], T["stat_detrend"]], index=0)
use_bessel = st.sidebar.checkbox("Bessel Filter", value=True)
cutoff = st.sidebar.slider(T["cutoff"], 100, int(st.session_state.fs_nyquist), 2000)

st.sidebar.header(T["sb_detec"])
threshold = st.sidebar.slider(T["threshold"], 1.0, 8.0, 2.5)

st.sidebar.header(T["sb_kinetics"])
use_amp_filter = st.sidebar.checkbox("Filter Amplitude", value=True)
amp_limit = st.sidebar.number_input(T["amp_filter"], min_value=0.0, value=7.0, step=1.0)

use_decay_filter = st.sidebar.checkbox("Filter Decay", value=True)
decay_limit = st.sidebar.number_input(T["decay_thresh"], value=4.0, step=0.5)

use_rise_filter = st.sidebar.checkbox("Filter Rise Time", value=True)
rise_limit = st.sidebar.number_input(T["rise_thresh"], value=0.5, step=0.1)

st.sidebar.header(T["sb_viz"])
y_zoom = st.sidebar.slider(T["zoom_y"], -300, 100, (-80, 20))

auto_z = st.sidebar.checkbox(T["auto_z"], value=True)

# Navigation Temporelle (Boutons + Inputs)
st.sidebar.markdown("**Navigation Temporelle X (s)**")
col_b1, col_b2 = st.sidebar.columns(2)
col_b1.button(T["btn_left"], on_click=scroll_left, use_container_width=True)
col_b2.button(T["btn_right"], on_click=scroll_right, use_container_width=True)

col_x1, col_x2 = st.sidebar.columns(2)
col_x1.number_input(T["x_start"], step=0.1, key="x_start")
col_x2.number_input(T["x_end"], step=0.1, key="x_end")

if st.session_state.x_start >= st.session_state.x_end:
    st.sidebar.error("Le début doit être inférieur à la fin.")
x_zoom = (st.session_state.x_start, st.session_state.x_end)

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

# --- ANALYSE ---
file = st.file_uploader("Charger .abf", type=["abf"])

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.abf') as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        abf = pyabf.ABF(tmp_path)
        abf.setSweep(0)
        fs, times, dt = abf.dataRate, abf.sweepX, 1000/abf.dataRate
        st.session_state.fs_nyquist = fs / 2

        # Ligne de base
        if baseline_mode == T["dyn_detrend"]:
            raw_data = ndimage.median_filter(abf.sweepY, size=int(0.5 * fs))
            raw_data = abf.sweepY - raw_data
        else:
            raw_data = abf.sweepY - np.median(abf.sweepY)

        # Filtrage
        f_data = raw_data
        if use_bessel:
            nyq = 0.5 * fs
            b, a = signal.bessel(4, cutoff/nyq, btype='low', analog=False)
            f_data = signal.filtfilt(b, a, raw_data)

        # Détection EPSC (Toujours Inward, on inverse le signal)
        detect_trace = -f_data
        
        best_corr = np.zeros_like(detect_trace)
        
        # Constantes de temps multi-échelles spécifiques à l'AMPA
        default_decays = [2.0, 5.0, 10.0, 15.0]
        
        for d in default_decays:
            t_tmpl = np.arange(0, 20, dt)
            tmpl = (np.exp(-t_tmpl/d) - np.exp(-t_tmpl/0.5)) 
            tmpl /= np.max(np.abs(tmpl))
            best_corr = np.maximum(best_corr, signal.correlate(detect_trace, tmpl, mode='same'))
            
        corr_z = (best_corr - np.mean(best_corr)) / np.std(best_corr)
        peaks, _ = signal.find_peaks(corr_z, height=threshold, distance=int(0.005 * fs))
        
        valid_ev = []
        k_trace = f_data
        
        for i, p in enumerate(peaks):
            start, end = p - int(0.003*fs), p + int(0.020*fs)
            if start < 0 or end >= len(k_trace): continue
            
            l_base = np.mean(k_trace[p-int(0.005*fs):p-int(0.002*fs)])
            
            # Extraction du segment (Inward -> inversion pour l'analyse)
            seg = -(k_trace[start:end] - l_base)
            
            amp = np.max(seg)
            rise_1090 = calculate_rise_time_expert(seg, dt)
            area = integrate.trapezoid(seg, dx=dt)
            
            estimated_decay = abs(area / amp) if amp > 0 else 0
            
            pass_amp = (not use_amp_filter or amp >= amp_limit)
            pass_decay = (not use_decay_filter or estimated_decay <= decay_limit)
            pass_rise = (not use_rise_filter or rise_1090 <= rise_limit)
            
            if pass_amp and pass_decay and pass_rise:
                ev = {'idx': p, 'time': times[p], 'amp': amp, 'rise': rise_1090, 'area': abs(area), 'decay': estimated_decay}
                ev['iei'] = (times[p] - times[peaks[i-1]])*1000 if i>0 else np.nan
                valid_ev.append(ev)

        # --- PLOTTING ---
        st.subheader(T["viz_header"])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[2,1]})
        
        ax1.plot(times, f_data, color='black', lw=0.5)
        
        if valid_ev:
            ax1.plot([e['time'] for e in valid_ev], [f_data[e['idx']] for e in valid_ev], 'o', color='#FF8C00', markersize=5)

        ax1.set_ylim(y_zoom)
        ax1.set_xlim(x_zoom)
        ax1.set_ylabel("Amplitude (pA)")

        ax2.plot(times, corr_z, color='blue', alpha=0.6)
        ax2.axhline(threshold, color='red', ls='--')
        ax2.set_ylabel("Z-Score")
        
        if auto_z:
            mask = (times >= st.session_state.x_start) & (times <= st.session_state.x_end)
            if np.any(mask):
                z_local = corr_z[mask]
                z_min, z_max = np.min(z_local), np.max(z_local)
                margin = abs(z_max - z_min) * 0.15 if z_max != z_min else 1.0
                ax2.set_ylim(z_min - margin, z_max + margin)

        st.pyplot(fig)
        
        # --- EXPORT & POPULATION ANALYSIS ---
        if valid_ev:
            df = pd.DataFrame(valid_ev)
            st.divider()
            
            freq_hz = len(df) / times[-1]
            mean_iei_ms = df['iei'].mean()
            
            st.subheader(f"Statistiques Globales | {len(valid_ev)} Événements détectés")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fréquence Moyenne", f"{freq_hz:.2f} Hz")
            c2.metric("Amplitude Moyenne", f"{df['amp'].mean():.2f} pA")
            c3.metric("Rise Time Moyen", f"{df['rise'].mean():.2f} ms")
            c4.metric("Decay Estimé Moyen", f"{df['decay'].mean():.2f} ms")
            
            col_exp1, col_exp2 = st.columns(2)
            
            # Export Events
            df_export = df[['time', 'amp', 'rise', 'decay', 'area', 'iei']].copy()
            csv_events = df_export.to_csv(index=False).encode('utf-8')
            col_exp1.download_button(label="📁 Télécharger Événements (CSV)", data=csv_events, file_name='sEPSC_events.csv', mime='text/csv')
            
            # Export Distributions
            n_bins = 25
            counts_amp, bins_amp = np.histogram(df['amp'], bins=n_bins)
            counts_rise, bins_rise = np.histogram(df['rise'], bins=n_bins)
            iei_clean = df['iei'].dropna()
            counts_iei, bins_iei = np.histogram(iei_clean, bins=n_bins) if not iei_clean.empty else (np.zeros(n_bins), np.zeros(n_bins+1))

            df_export_summary = pd.DataFrame({
                'Amp_Bin_Center_pA': (bins_amp[:-1] + bins_amp[1:]) / 2,
                'Amp_Counts': counts_amp,
                'Rise_Bin_Center_ms': (bins_rise[:-1] + bins_rise[1:]) / 2,
                'Rise_Counts': counts_rise,
                'IEI_Bin_Center_ms': (bins_iei[:-1] + bins_iei[1:]) / 2,
                'IEI_Counts': counts_iei
            })

            csv_summary = df_export_summary.to_csv(index=False).encode('utf-8')
            col_exp2.download_button(label="📊 Télécharger Distributions (CSV)", data=csv_summary, file_name='sEPSC_distributions.csv', mime='text/csv')

            # Figures
            fig2, (ha, hb, hc) = plt.subplots(1, 3, figsize=(15, 4))
            ha.bar((bins_amp[:-1] + bins_amp[1:]) / 2, counts_amp, width=(bins_amp[1]-bins_amp[0])*0.9, color='gray')
            ha.set_title("Amplitude (pA)")
            hb.bar((bins_rise[:-1] + bins_rise[1:]) / 2, counts_rise, width=(bins_rise[1]-bins_rise[0])*0.9, color='#FF8C00')
            hb.set_title("Rise Time 10-90% (ms)")
            if not iei_clean.empty:
                hc.bar((bins_iei[:-1] + bins_iei[1:]) / 2, counts_iei, width=(bins_iei[1]-bins_iei[0])*0.9, color='salmon')
                hc.set_title("IEI (ms)")
            st.pyplot(fig2)

    except Exception as e: st.error(f"Erreur d'analyse: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
