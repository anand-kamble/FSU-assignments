"""
utils.py: Utility functions for Interpolation Methods for Audio Upsampling project.

Description:
    This module provides utility functions for the Interpolation Methods for Audio Upsampling project.

Usage:
    - Use plotAudios to visualize and compare two audio signals with different sample rates.
    - Use StrechAudio to stretch the audio by repeating each element 'n' times.
    - Use Normalized to normalize the values in an input array to the range [0, 1].
    - Use Calculate_Error to calculate the Root Mean Square Error (RMSE) between two audio signals.

Dependencies:
    - NumPy
    - Matplotlib

Author:
    Anand Kamble

Date:
    2023-12-14
"""

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np


def plotAudios(a1: Tuple[int, np.ndarray], a2: Tuple[int, np.ndarray], start: int = 0, end: int = 0, plt=plt) -> None:
    """
    Plot two audio signals with different sample rates.

    Parameters:
    - audio1 (tuple): Tuple representing the first audio signal, where 'audio1[0]' is the sample rate and 'audio1[1]' is the audio data.
    - audio2 (tuple): Tuple representing the second audio signal, where 'audio2[0]' is the sample rate and 'audio2[1]' is the audio data.
    - start (int): Starting index for the plot. Default is 0.
    - end (int): Ending index for the plot. Default is 0 (end of the audio).
    - plt (module): The matplotlib.pyplot module for plotting. Default is matplotlib.pyplot.

    Returns:
    - None
    """
    lower_res = a1 if a1[0] < a2[0] else a2
    higher_res = a1 if a1[0] > a2[0] else a2

    L = [i for i in lower_res[1] for j in range(higher_res[0] // lower_res[0])]
    H = higher_res[1]

    if end <= start or end > len(H):
        end = len(H)

    plt.step(np.arange(end - start), L[start:end],
             label=f"Sample Rate-{lower_res[0]}")
    plt.step(np.arange(end - start), H[start:end],
             label=f"Sample Rate-{higher_res[0]}")
    plt.legend()


def StrechAudio(audio: Tuple[int, np.ndarray], n: int) -> np.ndarray:
    """
    Stretch the audio by repeating each element 'n' times.

    Parameters:
    - audio (list): A list containing audio data where 'audio[1]' represents the audio samples.
    - n (int): The factor by which the audio will be stretched.

    Returns:
    - list: Stretched audio data.
    """
    return [i for i in audio[1] for j in range(n)]


def Normalized(arr: np.ndarray) -> np.ndarray:
    """
    Normalize the values in the input array to the range [0, 1].

    Parameters:
    - arr (numpy.ndarray): Input array to be normalized.

    Returns:
    - numpy.ndarray: Normalized array.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)

    normalized_arr = (arr - min_val) / (max_val - min_val)

    return normalized_arr


def Calculate_Error(upscaled_audio: Tuple[int, np.ndarray], original_audio: Tuple[int, np.ndarray]) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between two audio signals.

    Parameters:
    - upscaled_audio (numpy.ndarray): Array containing the values of the upscaled audio signal.
    - original_audio (numpy.ndarray): Array containing the values of the original audio signal.

    Returns:
    - float: Root Mean Square Error (RMSE) between the two audio signals.
    """
    np.seterr(all='ignore')
    rmse = np.sqrt(np.mean(((upscaled_audio[1] - original_audio[1]) ** 2)))
    value_range = np.abs(np.max(original_audio[1]) - np.min(original_audio[1]))
    return (rmse / value_range) * 100
