import librosa
import numpy as np
from scipy.signal import freqz

import os, sys
import argparse

def lpmcc(audio: np.ndarray, sr:int, **kwargs):
    order = 12  # 線形予測次数
    FPS = 100 # LPCスペクトル包絡を何msごとに計算するか
    frame_length = sr//FPS # 線形予測する信号長（フレーム長）
    length = audio.shape[0]
    n_frame = length // frame_length
    n_fft = 2048
    worN = 2048//2
    n_mels = 24
    envelope = np.zeros((worN, n_frame))
    lpmcc = np.zeros((n_mels, n_frame))
    eps = 1e-3

    for k in range(n_frame):
        # 各時刻のlpcスペクトルの計算
        # LPCによるフィルタ次数計算
        a = librosa.lpc(audio, order)

        # 周波数応答
        freqs, h = freqz(1.0, a, worN=worN)
        #
        envelope[:, k] = np.abs(h)
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels=n_mels)
        lpmcc[:, k] = np.einsum("...ft,mf->...mt", envelope[:,k] ,mel_basis, optimize=True)

    return lpmcc


if __name__ == "__main__":
    # parsing args from command line 
    
