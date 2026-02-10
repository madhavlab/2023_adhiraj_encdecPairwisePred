import os
import torch
import torchaudio
import torchaudio.transforms as T

def read_audio(filepath, fs=16000, normalize=False, preemphasis=False):
    """
    Reads audio file stored at <filepath>
    Parameters:
        filepath (str): audio file path
        fs (int, optional): samping rate
        mono (boolean, optional): return single channel
        normalize(boolean, optional): peak normalization of signal
        preemphasis (boolean, optional): apply pre-emphasis filter
    Returns:
        waveform (tensor): audio signal, dim(N,)
    """
    assert isinstance(filepath, str), "filepath must be specified as string"
    assert os.path.exists(filepath), f"{filepath} does not exist."

    try:
        waveform, sr = torchaudio.load(filepath)
        if waveform.dim() == 2:
            waveform = waveform.squeeze_()

        # preemphasis
        if preemphasis:
            waveform = pre_emphasis(waveform)
        # resample
        if sr != fs:
            resampler = T.Resample(sr, fs, dtype=waveform.dtype)
            waveform = resampler(waveform)
        # normalize
        if normalize:
            waveform = rms_normalize(waveform)
        return waveform
    except Exception as e: 
        return None

def rms_normalize(waveform, r=-10):
    """
    RMS-normalization of  <waveform>
    Parameter:
        waveform (tensor): waveform, dims: (N,)
        rms (float): rms in dB
    """
    current_rms = torch.pow(torch.mean(torch.pow(waveform,2)) ,0.5)
    scaling_factor = (10**(r/10))/current_rms
    return waveform*scaling_factor


def pre_emphasis(waveform, coeff=0.97):
    filtered_sig = torch.empty_like(waveform)
    filtered_sig[1:] = waveform[1:] - coeff*waveform[:-1]
    filtered_sig[0] = waveform[0]
    return filtered_sig



