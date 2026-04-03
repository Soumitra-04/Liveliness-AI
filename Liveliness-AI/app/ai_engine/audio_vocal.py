import os, warnings 
from typing import List, Tuple
import numpy as np

# ── Backend selection ─────────────────────────────────────────────────────────

try:
    import librosa  # type: ignore
    _BACKEND = "librosa"
except ImportError:  # pragma: no cover
    librosa = None   # type: ignore
    _BACKEND = "scipy"

# ── Public API ────────────────────────────────────────────────────────────────

def process_audio(file_path: str) -> Tuple[float, str]:
    """
    Main entry point.
    Returns (score, explanation) where score 0=human, 1=synthetic.
    """
    # Load

    try:

        y = load_audio(file_path)
    except FileNotFoundError:
        return 1.0, f"❌ File not found: '{file_path}'"
    except Exception as e:
        return 1.0, f"❌ Load error: {e}"

    dur_s  = len(y) / SR
    chunks = make_chunks(y)
    n = len(chunks)


    # Ensemble scoring
    if LOADED_MODELS:
        combined    = np.zeros(n, dtype=np.float32)
        models_used = []
        for entry in LOADED_MODELS:
            try:
                cs = score_chunks(chunks, entry)

                combined   += entry['weight'] * np.array(cs)

                models_used.append(entry['id'].split('/')[0])

            except Exception as e:

                print(f'⚠️  {entry["id"]} failed: {e}')



        if models_used:

            score     = round(float(np.clip(np.percentile(combined, 80), 0, 1)), 4)

            max_chunk = round(float(np.max(combined)), 4)

            conf      = int(max(score, 1-score) * 100)

            info      = ' + '.join(models_used)



            if score < BAND_HUMAN:

                expl = (f'[REAL] {conf}% confidence — all {n} windows '

                        f'({dur_s:.1f}s) show human speech patterns. '

                        f'Models: {info}')

            elif score < BAND_UNCERTAIN:

                expl = (f'[UNCERTAIN] Score={score:.3f}. Mixed signals across '

                        f'{n} windows (peak={int(max_chunk*100)}%). Try a '

                        'longer, cleaner clip with continuous speech. '

                        f'Models: {info}')

            else:

                expl = (f'[FAKE] {conf}% confidence — synthetic artifacts '

                        f'detected across {n} windows '

                        f'(peak window={int(max_chunk*100)}%, {dur_s:.1f}s audio). '

                        f'Models: {info}')

            return score, expl



    # Librosa fallback

    c_scores = []

    def norm(v,lo,hi,inv=True):

        if hi<=lo: return 0.0

        r=float(np.clip((v-lo)/(hi-lo),0,1)); return 1-r if inv else r

    for chunk in chunks:

        mfcc  = librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=20)

        delta = librosa.feature.delta(mfcc)

        S_db  = librosa.power_to_db(librosa.feature.melspectrogram(y=chunk,sr=SR,n_mels=64),ref=np.max)

        rms   = librosa.feature.rms(y=chunk)[0]

        rs    = float(np.std(librosa.feature.spectral_rolloff(y=chunk,sr=SR,roll_percent=0.85)))

        cs = (0.25*norm(float(np.mean(np.var(mfcc,axis=1))),10,120)

             +0.20*norm(float(np.mean(np.var(delta,axis=1))),5,60)

             +0.25*norm(float(np.mean(np.abs(np.diff(S_db,axis=1)))),0.05,0.30)

             +0.15*norm(float(np.var(np.log1p(rms))),0.003,0.040)

             +0.15*norm(rs,80,600))

        c_scores.append(float(np.clip(cs,0,1)))

    score = round(float(np.percentile(c_scores,80)),4)

    return score, '⚠️  Librosa-only (no torch). Install torch+transformers for accuracy. Score='+str(score)

analyse_audio = process_audio

print('✅ Analysis engine ready.')

uploaded = files.upload()

if uploaded:

    audio_path = list(uploaded.keys())[0]

    print(f'✅ Ready: {audio_path}')

if 'audio_path' not in dir(): print('⚠️  Select a file first.')

else:

    print(f'📂 {audio_path}  ({os.path.getsize(audio_path)/1024/1024:.2f} MB)')

    display(ipd.Audio(audio_path))



if 'audio_path' not in dir():

    print('⚠️  Select a file first.')

else:

    fname = os.path.basename(audio_path)

    print(f'🔍 Analysing: {fname}  ({len(LOADED_MODELS)} models in ensemble)\n')



    score, explanation = process_audio(audio_path)



    BAR    = 40

    filled = int(round(score * BAR))

    bar    = '█'*filled + '░'*(BAR-filled)



    if score < BAND_HUMAN:

        verdict, col = '🟢 LIKELY HUMAN',              '\033[92m'

    elif score < BAND_UNCERTAIN:

        verdict, col = '🟡 UNCERTAIN',                 '\033[93m'

    else:

        verdict, col = '🔴 LIKELY SYNTHETIC/DEEPFAKE', '\033[91m'

    R = '\033[0m'

    print('='*62)
    print(f'  FILE       : {fname}')
    print(f'  SCORE      : {col}{score:.4f}{R}  [{bar}]')
    print(f'  VERDICT    : {col}{verdict}{R}')
    print(f'  DETAIL     : {explanation}')
    print('='*62)



if 'audio_path' not in dir():
    print('⚠️  Select a file first.')

else:
    y = _load_audio(audio_path)
    flux      = _spectral_flux(y)
    rms_var   = _rms_variance(y)
    mfcc_tvar = _mfcc_temporal_variance(y)
    hnr_db    = _harmonics_to_noise(y)
    r_std     = _rolloff_std(y)
    s_conf    = _speech_confidence(y)


    print('Raw feature values (useful for understanding the score):\n')
    print(f'  Spectral Flux          : {flux:.4f}   (expect 0.05–0.30 for real speech)')
    print(f'  RMS Energy Variance    : {rms_var:.5f}  (expect 0.003–0.040 for real speech)')
    print(f'  MFCC Temporal Variance : {mfcc_tvar:.2f}   (expect 10–120 for real speech)')
    print(f'  Harmonics-to-Noise     : {hnr_db:.2f} dB  (suspicious if > 28 dB)')
    print(f'  Rolloff Std            : {r_std:.1f} Hz  (expect 80–600 Hz for real speech)')
    print(f'  Speech Confidence Gate : {s_conf:.3f}    (0=not speech, 1=clear speech)')