import librosa
import librosa.display
import numpy as np
from scipy.signal import freqz

import os, sys
import argparse
from tqdm import tqdm

from matplotlib import pylab as plt

def lpmcc(audio: np.ndarray, sr:int, **kwargs):
    order = 12  # 線形予測次数
    FPS = 100 # LPCスペクトル包絡を何msごとに計算するか
    frame_length = sr//FPS # 線形予測する信号長（フレーム長）
    length = audio.shape[0] # 音の長さ
    n_frame = length // frame_length # 何フレームあるか
    n_fft = 1024
    worN = 1024//2 + 1
    n_mels = 24
    envelope = np.zeros((worN, n_frame))
    lpmcc = np.zeros((n_mels, n_frame))
    # eps = 1e-3

    print(audio.shape[0])

    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    print(n_frame, frame_length)
    for k in tqdm(range(n_frame)):
        # 各時刻のlpcスペクトルの計算
        # LPCによるフィルタ次数計算
        slc = slice(k*frame_length, (k+1)*frame_length)
        a = librosa.lpc(audio[slc], order=order)

        # フィルタの周波数インデックスおよび周波数応答（）
        freqs, h = freqz(1.0, a, worN=worN)
        
        # 
        envelope[:, k] = np.abs(h)
        
        # print(mel_basis.shape, envelope[:,k].shape)
        lpmcc[:, k] = np.log10(np.dot(mel_basis, envelope[:,k]))

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(lpmcc, sr=sr, x_axis='time', cmap='turbo')
    plt.colorbar()
    plt.show()
    print(lpmcc.shape)
    print(envelope.shape)

    return lpmcc


if __name__ == "__main__":
    # parsing args from command line 
    parser = argparse.ArgumentParser(description='LPMCC Audio Processing')
    parser.add_argument('--audiofile', type=str, help='Path to the audio file')
    parser.add_argument('--outputfile', type=str, help='Path to save the output file')
    args = parser.parse_args()
    
    # Load audio file
    audio, sr = librosa.load(args.audiofile, sr=None, mono=True)
    
    # Call lpmcc function
    lpmcc_result = lpmcc(audio, sr)
    
    # Save output file
    np.save(args.outputfile, lpmcc_result)