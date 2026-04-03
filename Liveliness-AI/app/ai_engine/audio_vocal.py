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

