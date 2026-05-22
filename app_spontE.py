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

root_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(root_dir) == "pages":
    root_dir = os.path.dirname(root_dir)
logo_path = os.path.join(root_dir, "logo_chavis_final.png")

if 'fs_nyquist' not in st.session_state:
    st.session_state.fs_nyquist = 5000.0
if 'x_start' not in st.session_state:
    st.session_state.x_start = 10.0
if 'x_end' not in st.session_state:
    st.session_state.x_end = 11.0

# --- FONCTIONS MATHÉMATIQUES EXPERTES ---
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

def robust_z_score(sig):
    """Z-score insensible aux très grands artefacts isolés."""
    med = np.median(sig)
    mad = np.median(np.abs(sig - med))
    if mad == 0:
        return (sig - np.mean(sig)) / (np.std(sig) + 1e-9)
    return (sig - med) / (1.4826 * mad)

def get_true_peaks(corr_peaks, detect_trace, search_window, fs):
    """
    CORRECTION MAJEURE : Snapping Biologique.
    La corrélation a un décalage de phase. Cette fonction utilise les index de corrélation 
    pour chercher le véritable pic biologique local dans le signal lissé.
    """
    if len(corr_peaks) == 0: return []
    true_peaks = []
    for cp in corr_peaks:
        start_search = max(0, cp - search_window)
        end_search = min(len(detect_trace), cp + search_window)
        
        # Trouve le vrai pic (maximum) dans cette petite fenêtre
        local_max_idx = start_search + np.argmax(detect_trace[start_search:end_search])
        true_peaks.append(local_max_idx)
        
    # Élimine les doublons si deux pics de corrélation convergent vers le même pic biologique
    true_peaks = sorted(list(set(true_peaks)))
    filtered_peaks = [true_peaks[0]]
    for p in true_peaks[1:]:
        if p - filtered_peaks[-1] > int(0.002 * fs): # Période réfractaire 2ms
            filtered_peaks.append(p)
            
    return filtered_peaks

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

# --- INTERFACE ---
lang = st.sidebar.selectbox("Language / Langue", ["Français", "English"])
T = {
    "Français": {
        "title": "sEPSC : Template Matching Itératif",
        "subtitle": "Double passe avec recentrage biologique (Snapping) et Template Scaling.",
    },
    "English": {
        "title": "sEPSC: Iterative Template Matching",
        "subtitle": "Double pass with biological peak snapping and Template Scaling.",
    }
}[lang]

col_logo, col_title = st.columns([1, 4])
with col_logo:
    if os.path.exists(logo_path): st.image(logo_path, width=150)
    else: st.write("🟢")
with col_title:
    st.title(f"🟢 {T['title']}")
    st.markdown(f"*{T['subtitle']}*")
st.divider()

st.sidebar.header("1. Prétraitement (AMPA)")
baseline_mode = st.sidebar.radio("Ligne de base", ["Detrending Dynamique (Médiane)", "Médiane Globale Statique"], index=0)
use_bessel = st.sidebar.checkbox("Bessel Filter", value=True)
cutoff = st.sidebar.slider("Coupure Bessel (Hz)", 100, int(st.session_state.fs_nyquist), 2000)

st.sidebar.header("2. Seuil de Détection")
threshold = st.sidebar.slider("Seuil Z-Score (Robuste)", 1.0, 8.0, 3.0)

st.sidebar.header("3. Filtres de Base (Tier 1)")
st.sidebar.caption("Sert à construire l'empreinte cellulaire. Soyez assez strict ici.")
use_amp_filter = st.sidebar.checkbox("Filtrer Amplitude", value=True)
amp_limit = st.sidebar.number_input("Amplitude Min (pA)", min_value=0.0, value=10.0, step=1.0)
use_rise_filter = st.sidebar.checkbox("Filtrer Rise Time", value=True)
rise_limit = st.sidebar.number_input("Rise Time Max (ms)", value=4.0, step=0.1)

st.sidebar.header("4. Visualisation")
y_zoom = st.sidebar.slider("Zoom Y (pA)", -300, 100, (-80, 20))
auto_z = st.sidebar.checkbox("Auto-ajustement axe Z", value=True)

col_b1, col_b2 = st.sidebar.columns(2)
col_b1.button("⬅️ Gauche", on_click=scroll_left, use_container_width=True)
col_b2.button("Droite ➡️", on_click=scroll_right, use_container_width=True)
col_x1, col_x2 = st.sidebar.columns(2)
col_x1.number_input("Début (s)", step=0.1, key="x_start")
col_x2.number_input("Fin (s)", step=0.1, key="x_end")
x_zoom = (st.session_state.x_start, st.session_state.x_end)

# --- ANALYSE ---
file = st.file_uploader("Charger .abf / Upload .abf", type=["abf"])

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.abf') as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        abf = pyabf.ABF(tmp_path)
        abf.setSweep(0)
        fs, times, dt = abf.dataRate, abf.sweepX, 1000/abf.dataRate
        st.session_state.fs_nyquist = fs / 2

        if baseline_mode == "Detrending Dynamique (Médiane)":
            raw_data = ndimage.median_filter(abf.sweepY, size=int(0.5 * fs))
            raw_data = abf.sweepY - raw_data
        else:
            raw_data = abf.sweepY - np.median(abf.sweepY)

        f_data = raw_data
        if use_bessel:
            nyq = 0.5 * fs
            b, a = signal.bessel(4, cutoff/nyq, btype='low', analog=False)
            f_data = signal.filtfilt(b, a, raw_data)

        detect_trace = -f_data # Inversion (Inward EPSC -> Pics positifs)
        
        # ==========================================
        # PASSE 1 : MATHEMATICAL TEMPLATE (BASE)
        # ==========================================
        best_corr_base = np.zeros_like(detect_trace)
        default_decays = [2.0, 5.0, 10.0, 15.0]
        
        for d in default_decays:
            t_tmpl = np.arange(0, 20, dt)
            tmpl = (np.exp(-t_tmpl/d) - np.exp(-t_tmpl/0.5)) 
            tmpl /= np.max(np.abs(tmpl))
            best_corr_base = np.maximum(best_corr_base, signal.correlate(detect_trace, tmpl, mode='same'))
            
        corr_z_base = robust_z_score(best_corr_base)
        peaks_base_corr, _ = signal.find_peaks(corr_z_base, height=threshold, distance=int(0.005 * fs))
        
        # SNAP BIOLOGIQUE (Réalignement)
        peaks_base = get_true_peaks(peaks_base_corr, detect_trace, search_window=int(0.010 * fs), fs=fs)
        
        valid_ev_base = []
        window_pre = int(0.005 * fs)
        window_post = int(0.025 * fs)
        extracted_waveforms = []

        for i, p in enumerate(peaks_base):
            start, end = p - window_pre, p + window_post
            if start < 0 or end >= len(f_data): continue
            
            l_base = np.mean(f_data[p-window_pre:p-int(0.002*fs)])
            seg = -(f_data[start:end] - l_base)
            
            amp = seg[window_pre] # Le vrai pic est exactement à window_pre grâce au snapping !
            rise_1090 = calculate_rise_time_expert(seg, dt)
            area = integrate.trapezoid(seg, dx=dt)
            estimated_decay = abs(area / amp) if amp > 0 else 0
            
            pass_amp = (not use_amp_filter or amp >= amp_limit)
            pass_rise = (not use_rise_filter or rise_1090 <= rise_limit)
            
            if pass_amp and pass_rise:
                ev = {'idx': p, 'time': times[p], 'amp_peak': amp, 'rise': rise_1090, 'area': abs(area), 'decay': estimated_decay}
                ev['iei'] = (times[p] - times[peaks_base[i-1]])*1000 if len(valid_ev_base)>0 else np.nan
                valid_ev_base.append(ev)
                extracted_waveforms.append(seg)

        # ==========================================
        # CREATION DE L'EMPREINTE & PASSE 2 (ITÉRATIVE)
        # ==========================================
        valid_ev_iter = []
        
        if len(extracted_waveforms) > 5:
            # Empreinte par Médiane Parfaite (Puisque tous les pics sont alignés au centre)
            avg_waveform = np.median(extracted_waveforms, axis=0)
            avg_waveform -= np.mean(avg_waveform[:int(0.002*fs)])
            avg_waveform = np.clip(avg_waveform, 0, None)
            
            # Normalisation
            if np.max(avg_waveform) > 0:
                avg_waveform /= np.max(avg_waveform) 
            
            # PASSE 2
            corr_iter = signal.correlate(detect_trace, avg_waveform, mode='same')
            corr_z_iter = robust_z_score(corr_iter)
            
            peaks_iter_corr, _ = signal.find_peaks(corr_z_iter, height=threshold, distance=int(0.005 * fs))
            
            # SNAP BIOLOGIQUE POUR LA PASSE 2
            peaks_iter = get_true_peaks(peaks_iter_corr, detect_trace, search_window=int(0.010 * fs), fs=fs)
            
            for i, p in enumerate(peaks_iter):
                start, end = p - window_pre, p + window_post
                if start < 0 or end >= len(f_data): continue
                
                l_base = np.mean(f_data[p-window_pre:p-int(0.002*fs)])
                seg = -(f_data[start:end] - l_base)
                
                # TEMPLATE SCALING MATH
                scale_factor = np.dot(seg, avg_waveform) / (np.dot(avg_waveform, avg_waveform) + 1e-9)
                amp_scaled = scale_factor 
                amp_peak = seg[window_pre]
                
                rise_1090 = calculate_rise_time_expert(seg, dt)
                
                # En passe 2, on récupère tout ce qui matche, sans filtre strict sur le Rise Time
                if (not use_amp_filter or amp_scaled >= (amp_limit * 0.75)): # Tolérance pour les minis distaux
                    ev = {'idx': p, 'time': times[p], 'amp_scaled': amp_scaled, 'amp_peak': amp_peak, 'rise': rise_1090}
                    ev['iei'] = (times[p] - times[peaks_iter[i-1]])*1000 if len(valid_ev_iter)>0 else np.nan
                    valid_ev_iter.append(ev)

            # --- PLOTTING ---
            st.success(f"**Analyse Réussie !** Empreinte créée avec {len(valid_ev_base)} EPSC parfaits. **{len(valid_ev_iter)} EPSC totaux détectés.**")
                
            col_graph1, col_graph2 = st.columns([3, 1])
            
            with col_graph1:
                st.subheader("Trace & Détections Itératives")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios':[2,1]})
                
                ax1.plot(times, f_data, color='black', lw=0.5)
                # Les points orange marquent les événements finaux de la Passe 2
                ax1.plot([e['time'] for e in valid_ev_iter], [f_data[e['idx']] for e in valid_ev_iter], 'o', color='#FF8C00', markersize=5, label='Iterative Hits')
                ax1.set_ylim(y_zoom)
                ax1.set_xlim(x_zoom)
                ax1.set_ylabel("Amplitude (pA)")

                ax2.plot(times, corr_z_iter, color='blue', alpha=0.6)
                ax2.axhline(threshold, color='red', ls='--')
                ax2.set_ylabel("Robust Z-Score")
                
                if auto_z:
                    mask = (times >= st.session_state.x_start) & (times <= st.session_state.x_end)
                    if np.any(mask):
                        z_local = corr_z_iter[mask]
                        z_min, z_max = np.min(z_local), np.max(z_local)
                        margin = abs(z_max - z_min) * 0.15 if z_max != z_min else 1.0
                        ax2.set_ylim(z_min - margin, z_max + margin)

                st.pyplot(fig)

            with col_graph2:
                st.subheader("Empreinte (Aligned)")
                fig_avg, ax_avg = plt.subplots(figsize=(4, 6))
                t_avg = np.arange(len(avg_waveform)) * dt
                ax_avg.plot(t_avg, -avg_waveform, color='red', lw=2)
                ax_avg.set_title(f"Médiane Alignée (n={len(extracted_waveforms)})")
                ax_avg.set_xlabel("Temps (ms)")
                ax_avg.set_ylabel("Normalisé")
                ax_avg.grid(True, alpha=0.3)
                st.pyplot(fig_avg)

        else:
            st.error("Pas assez d'événements clairs (<5) en Passe 1. Baissez le Seuil Z-Score ou relâchez le filtre d'Amplitude/Rise Time.")

        # --- EXPORT & HISTOGRAMMES ---
        if len(valid_ev_iter) > 0:
            df_iter = pd.DataFrame(valid_ev_iter)
            st.divider()
            
            freq_hz = len(df_iter) / times[-1]
            st.subheader(f"Statistiques Globales | {len(valid_ev_iter)} Événements")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Fréquence", f"{freq_hz:.2f} Hz")
            c2.metric("Amplitude Moyenne (Scaled)", f"{df_iter['amp_scaled'].mean():.2f} pA")
            c3.metric("Rise Time Moyen", f"{df_iter['rise'].mean():.2f} ms")
            
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            
            df_base = pd.DataFrame(valid_ev_base)
            csv_base = df_base[['time', 'amp_peak', 'rise', 'decay', 'area', 'iei']].to_csv(index=False).encode('utf-8')
            col_exp1.download_button(label="📁 CSV - Base (Passe 1)", data=csv_base, file_name='sEPSC_Base.csv', mime='text/csv')

            csv_iter = df_iter[['time', 'amp_scaled', 'amp_peak', 'rise', 'iei']].to_csv(index=False).encode('utf-8')
            col_exp2.download_button(label="📁 CSV - Itératif (Passe 2)", data=csv_iter, file_name='sEPSC_Iterative.csv', mime='text/csv')
            
            n_bins = 25
            counts_amp, bins_amp = np.histogram(df_iter['amp_scaled'], bins=n_bins)
            counts_rise, bins_rise = np.histogram(df_iter['rise'], bins=n_bins)
            iei_clean = df_iter['iei'].dropna()
            counts_iei, bins_iei = np.histogram(iei_clean, bins=n_bins) if not iei_clean.empty else (np.zeros(n_bins), np.zeros(n_bins+1))

            df_export_summary = pd.DataFrame({
                'Amp_Bin_Center_pA': (bins_amp[:-1] + bins_amp[1:]) / 2, 'Amp_Counts': counts_amp,
                'Rise_Bin_Center_ms': (bins_rise[:-1] + bins_rise[1:]) / 2, 'Rise_Counts': counts_rise,
                'IEI_Bin_Center_ms': (bins_iei[:-1] + bins_iei[1:]) / 2, 'IEI_Counts': counts_iei
            })
            csv_summary = df_export_summary.to_csv(index=False).encode('utf-8')
            col_exp3.download_button(label="📊 CSV - Distributions", data=csv_summary, file_name='sEPSC_distributions.csv', mime='text/csv')

            fig2, (ha, hb, hc) = plt.subplots(1, 3, figsize=(15, 4))
            ha.bar((bins_amp[:-1] + bins_amp[1:]) / 2, counts_amp, width=(bins_amp[1]-bins_amp[0])*0.9, color='gray')
            ha.set_title("Amplitude Scaled (pA)")
            hb.bar((bins_rise[:-1] + bins_rise[1:]) / 2, counts_rise, width=(bins_rise[1]-bins_rise[0])*0.9, color='#FF8C00')
            hb.set_title("Rise Time 10-90% (ms)")
            if not iei_clean.empty:
                hc.bar((bins_iei[:-1] + bins_iei[1:]) / 2, counts_iei, width=(bins_iei[1]-bins_iei[0])*0.9, color='salmon')
                hc.set_title("IEI (ms)")
            st.pyplot(fig2)

    except Exception as e: 
        st.error(f"Erreur d'analyse: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
