import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def fft(df):
    fft_bf = np.array(df.values)

    N = len(fft_bf)

    #FFT
    F = np.fft.fft(fft_bf)
    # FFT結果（複素数）を絶対値に変換
    F_abs = np.abs(F)
    # 振幅を元に信号に揃える
    F_abs_amp = F_abs / N * 2 # 交流成分はデータ数で割って2倍する
    F_abs_amp[0] = F_abs_amp[0] / 2 # 直流成分（今回は扱わないけど）は2倍不要

    plt.plot(F_abs_amp[:int(N/2)+1])
    
    return F_abs_amp[:int(N/2)+1]
        
        