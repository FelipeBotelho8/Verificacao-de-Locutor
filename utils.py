from config import *
import numpy as np
import librosa

def get_mfcc_coefs(path):
    '''
    Função que carrega o áudio, dado o caminho, e calcula os coeficientes mel-cepstrais

    Input: path - caminho para o audio
    Output: vetor de coeficientes mel-cepstrais calculados do áudio 
    '''
    
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
    audio_emp = librosa.effects.preemphasis(audio)
    coefs = librosa.feature.mfcc(y=audio_emp, sr=SAMPLE_RATE, n_mfcc=MEL_ORDER, n_fft=FRAME_LENGTH, hop_length=160, n_mels='128') 
    if coefs.shape[1] != 300:
        coefs = librosa.util.fix_length(coefs, 300, axis=1)
    coefs = coefs.reshape(-1)

    return coefs