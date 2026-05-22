import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate, ndimage
import tempfile
import os
import pandas as pd

try:
    import pyabf
except ImportError:
    st.error("pyabf not found. Run: pip install pyabf")

# --- CONFIGURATION ---
st.set_page_config(page_title="Expert sEPSC Pipeline", layout="wide")

# Logo handling
logo_path = "logo_chavis_final.png" 

# --- FUNCTIONS ---
def robust_z_score(sig):
    med = np.median(sig)
    mad = np.median(np.abs(sig - med))
    if mad == 0: return (sig - np.mean(sig)) / (np.std(sig) + 1e-9)
    return (sig - med) / (1.4826 * mad)

def get_true_peaks(corr_peaks, detect_trace, search_window, fs):
    if len(corr_peaks) == 0: return []
    true_peaks = []
    for cp in corr_peaks:
        start_search = max(0, cp - search_window)
        end_search = min(len(detect_trace), cp + search_window)
        local_max_idx = start_search + np.argmax(detect_trace[start_search:end_search])
        true_peaks.append(local_max_idx)
    true_peaks = sorted(list(set(true_peaks)))
    filtered_peaks = [true_peaks[0]]
    for p in true_peaks[1:]:
        if p - filtered_peaks[-1] > int(0.002 * fs):
            filtered_peaks.append(p)
    return filtered_peaks

# --- INTERFACE ---
st.title("🟢 Expert sEPSC: Iterative Template Matching")
st.info("**DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19915015.svg)](https://doi.org/10.5281/zenodo.19915015) | [📖 Read README on GitHub](https://github.com/OliManzoni/Manzoni_Chavis_Lab_spontE)")

with st.expander("🔬 Methodology & Biophysical Rigor"):
    st.markdown("""
    * **Iterative Template Matching:** Uses a two-pass detection to build a cell-specific fingerprint, minimizing noise bias.
    * **Biological Snapping:** Corrects correlation phase shifts to align with true physiological peaks.
    * **Least Squares Template Scaling:** Provides robust amplitude and charge (fC) extraction.
    """)

# Sidebar
file = st.file_uploader("Upload .abf", type=["abf"])
threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 8.0, 3.0)

export_container = st.container()

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.abf') as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        abf = pyabf.ABF(tmp_path)
        abf.setSweep(0)
        fs, times, dt = abf.dataRate, abf.sweepX, 1000/abf.dataRate
        
        # Preprocessing
        f_data = abf.sweepY - np.median(abf.sweepY)
        detect_trace = -f_data 
        
        # Tier 1 Detection
        best_corr_base = np.zeros_like(detect_trace)
        for d in [2.0, 5.0, 10.0, 15.0]:
            t_tmpl = np.arange(0, 20, dt)
            tmpl = (np.exp(-t_tmpl/d) - np.exp(-t_tmpl/0.5)) 
            tmpl /= np.max(np.abs(tmpl))
            best_corr_base = np.maximum(best_corr_base, signal.correlate(detect_trace, tmpl, mode='same'))
        
        peaks_base = get_true_peaks(signal.find_peaks(robust_z_score(best_corr_base), height=threshold)[0], detect_trace, 10, fs)
        
        # Tier 2 (Iterative)
        extracted_waveforms = [-(f_data[p-int(0.005*fs):p+int(0.025*fs)] - np.mean(f_data[p-int(0.005*fs):p-int(0.002*fs)])) for p in peaks_base if p > int(0.005*fs)]
        
        if len(extracted_waveforms) > 5:
            avg_waveform = np.median(extracted_waveforms, axis=0)
            avg_waveform /= np.max(avg_waveform)
            template_area = integrate.trapezoid(avg_waveform, dx=dt)
            
            corr_iter = signal.correlate(detect_trace, avg_waveform, mode='same')
            peaks_iter = get_true_peaks(signal.find_peaks(robust_z_score(corr_iter), height=threshold)[0], detect_trace, 10, fs)
            
            valid_ev_iter = []
            for p in peaks_iter:
                seg = -(f_data[p-int(0.005*fs):p+int(0.025*fs)] - np.mean(f_data[p-int(0.005*fs):p-int(0.002*fs)]))
                scale = np.dot(seg, avg_waveform) / (np.dot(avg_waveform, avg_waveform) + 1e-9)
                valid_ev_iter.append({'time': times[p], 'amp': scale, 'charge': scale * template_area})
            
            st.success(f"Detected {len(valid_ev_iter)} EPSCs.")
            
            # Export
            with export_container:
                st.subheader("💾 Data Export")
                df = pd.DataFrame(valid_ev_iter)
                st.download_button("📁 Download Iterative Results (CSV)", df.to_csv(index=False), "sEPSC_Results.csv")
                
    except Exception as e: st.error(f"Analysis Error: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
