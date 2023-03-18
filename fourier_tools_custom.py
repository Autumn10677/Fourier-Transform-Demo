import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import io

from scipy import signal
from scipy.io import wavfile
from IPython.display import Audio
from ipywidgets import interactive
from tqdm import tqdm

def plot_waveform(amplitude=0.5, frequency=np.pi, phase_shift=np.pi):
    """
    Calculates the resulting waveform of two overlapping waves
    
    Parameters
    ----------
    amplitude : float
        Amplitude of the second wave
    frequency : float
        Frequency of the second wave
    phase_shift : float
        Phase shift of the second wave
        
    Returns
    -------
    """
    
    ### Calculates plotting data
    t = np.linspace(0, 4*np.pi, 1000)
    y1 = np.sin(t)
    y2 = amplitude * np.sin(frequency*t + phase_shift)
    summed_wave = y1 + y2

    ### Plotting
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.plot(t, y1, label='Wave 1', ls='--', lw=0.9)
    plt.plot(t, y2, label='Wave 2', ls='--', lw=0.9)
    plt.plot(t, summed_wave, label='Resulting Wave', lw=3)
    plt.axhline(0, color='k')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Strength')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
def sawtooth(t, sawtooth_period):
    """
    Simulates the waveform of a sawtooth wave.
    
    Parameters
    ----------
    t : list
        List of times our signal is simulated over
    sawtooth_period : float
        The period of our sawtooth wave in seconds
        
    Returns
    -------
    sawtooth_values : list
        The signal strength of our sawtooth wave at each time in t
    """
    
    return signal.sawtooth(2 * np.pi / sawtooth_period * t)

def fourier_coefficients(t, y, n):
    """
    Calculates the Fourier coefficients for our function
    
    Parameters
    ----------
    t : list
        List of times our signal is simulated over
    y : list
        Signal strength of our base function
    n : int
        Number of terms in our Fourier series
        
    Returns
    -------
    a0 : float
        First term in our Fourier series
    an : list
        Coefficients for all cosine terms
    bn : list
        Coefficients for all sine terms
    """
    
    ### Finds length of our signal and a0 coefficient
    T = t[-1] - t[0]
    a0 = 2*np.trapz(y, t)/T
    
    ### Initializes arrays for holding coefficient data
    an = np.zeros(n)
    bn = np.zeros(n)
    
    ### Calculates an and bn for each term n in our series
    for i in range(1, n+1):
        an[i-1] = 2/T*np.trapz(y*np.cos(2*np.pi*i*t/T), t)
        bn[i-1] = 2/T*np.trapz(y*np.sin(2*np.pi*i*t/T), t)
    
    return a0, an, bn

def fourier_series(t, a0, an, bn):
    """
    Calculates the Fourier series given the expansion coefficients
    
    Parameters
    ----------
    t : list
        List of times our signal is simulated over
    a0 : float
        First expansion coefficient of our Fourier series
    an : list
        Coefficients for all cosine terms
    bn : list
        Coefficients for all sine terms
        
    Returns
    -------
    fs : list
        Signal strength of our Fourier series at each time in t
    """
    
    ### Initialializes our Fourier series array with zeroes and a0
    fs = np.zeros_like(t)
    fs += a0/2

    ### Calculates expansion term values and adds them to fs
    for i in range(1, len(an)+1):
        fs += an[i-1]*np.cos(2*np.pi*i*t/(t[-1]-t[0]))
        fs += bn[i-1]*np.sin(2*np.pi*i*t/(t[-1]-t[0]))
    
    return fs

def plot_fourier(sawtooth_period, N):
    """
    Calculates a fourier series for a sawtooth wave up to N terms
    
    Parameters
    ----------
    sawtooth_period : float
        Period of our sawtooth wave in seconds
    N : int
        Number of terms in the Fourier series
        
    Returns
    -------
    """
    
    # Generate the sawtooth wave
    t = np.linspace(0, 5, 10000)
    y = sawtooth(t, sawtooth_period)

    # Calculate the Fourier coefficients
    a0, an, bn = fourier_coefficients(t, y, N)
    
    # Calculate the Fourier series
    fs = fourier_series(t, a0, an, bn)

    # Plot the results
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.plot(t, y, label='Original Signal')
    plt.plot(t, fs, label='Fourier Series')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.legend(loc='upper right')
    plt.show()
    
def jukebox_player(entry_number=0, max_clip_length=3., file_source=None):

    """
    Performs audio reconstruction using a user-specified audio with FFTs
    
    Parameters
    ----------
    entry_number : int
        Period of our sawtooth wave in seconds
    max_clip_length : float
        Maximum length of each clip at a given reconstruction
        
    Returns
    -------
    """
    
    catalogue = ["BabyElephantWalk60.wav", # 0
                 "CantinaBand3.wav",       # 1
                 "CantinaBand60.wav",      # 2
                 "Fanfare60.wav",          # 3
                 "gettysburg10.wav",       # 4
                 "gettysburg.wav",         # 5
                 "ImperialMarch60.wav",    # 6
                 "PinkPanther30.wav",      # 7
                 "PinkPanther60.wav",      # 8
                 "preamble10.wav",         # 9
                 "preamble.wav",           # 10
                 "StarWars3.wav",          # 11
                 "StarWars60.wav",         # 12
                 "taunt.wav"]              # 13
    
    ### Checks the entry number is valid
    if type(entry_number)!=int or entry_number<0 or entry_number>13:
        print("Invalid entry. Make sure your entry number is an integer between 0 and 13")
        return

    plt.rcParams["figure.figsize"] = (13, 8)
    
    ### Checks whether the user has provided a filesource
    if type(file_source)!=None and type(file_source)==str:
        
        ### Sets title with filename and extracts user audio
        plt.title("Reconstruction of " + file_source)
        sampling_rate, audio = wavfile.read(file_source)
    
    ### Loads Jukebox data
    else:
        
        ### Sets title with filename
        plt.title("Reconstruction of " + str(catalogue[entry_number]))
        
        ### Reads data for user-specified URL
        url = 'https://www2.cs.uic.edu/~i101/SoundFiles/' + catalogue[entry_number]
        with urllib.request.urlopen(url) as url_file:
            data = url_file.read()

        ### Extracts audio data
        bytes_io = io.BytesIO(data)
        sampling_rate, audio = wavfile.read(bytes_io)

    ### Perform the FFT and finds FFT magnitudes
    fft_y = np.fft.fft(audio[:int(sampling_rate*max_clip_length)])
    mag_fft_y = np.abs(fft_y)

    ### Calculate the frequencies associated with each coefficient
    freqs = np.fft.fftfreq(len(fft_y), 1/sampling_rate)
    
    ### Makes list of N terms to be added into audio reconstruction
    N_vals = [1]
    while N_vals[-1]<len(mag_fft_y):
        N_vals.append(N_vals[-1]*10)

    ### For each N-top terms, generates the reconstructed audio
    final_audio = np.array([])
    for N in tqdm(N_vals):

        ### Find the indices of the N largest magnitude FFT coefficients
        ### and sets all other coefficients to zero
        top_indices = np.argsort(mag_fft_y)[-N:]
        fft_y_filtered = np.zeros_like(fft_y)
        fft_y_filtered[top_indices] = fft_y[top_indices]

        ### Perform the inverse FFT to obtain the time-domain audio signal
        ifft_y = np.fft.ifft(fft_y_filtered)

        ### Normalize and appends the audio signal
        ifft_y /= np.max(np.abs(ifft_y))
        final_audio = np.append(final_audio, ifft_y)

    ### Plays the reconstructed audio
    gen_audio = Audio(np.real(final_audio), rate=sampling_rate)
    display(gen_audio)
    
    ### Scales the original audio to match the fitted data
    clipped_source = np.real(audio[:int(sampling_rate*max_clip_length)])
    amax = max(clipped_source)
    renormed_source = [i/amax for i in clipped_source]
    
    ### Creates original reference data for overplotting
    comp_audio = []
    for N in N_vals:
        comp_audio.extend(renormed_source)
    
    ### Generates time-coordinates for each point of our waveform
    audio_length = len(N_vals)*max_clip_length/len(final_audio)
    ts = [i*audio_length for i in range(len(final_audio))]
    
    ### Plots waveform of audio reconstruction
    plt.plot(ts, np.real(final_audio), label='reconstructed')
    plt.plot(ts, np.real(comp_audio), label='original', alpha=0.75)
    
    for i in range(len(N_vals)-1):
        plt.axvline(max_clip_length*(i+1), ls='dashed', color='black')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Waveform")
    plt.legend(loc='upper right')
    plt.show()
    
def apply_bandpass_filter(fshift, center_radius, outer_radius):
    
    """
    Applies a bandpass filter to an image's Fourier transform
    
    Parameters:
    -----------
    fshift : array
        The Fourier transformed image to be filtered
    center_radius : float
        Center radius of bandpass filter
    outer_radius : float
        Outer radius of bandpass filter
        
    Returns:
    --------
    fshift : array
        Filtered version of the input Fourier transformed image
    """
    
    # Get the number of rows and columns in the Fourier transformed image
    rows, cols = fshift.shape
    
    # Calculate the center of the Fourier transformed image
    crow, ccol = rows//2, cols//2
    
    # Create a grid of x and y values for the Fourier transformed image
    x, y = np.meshgrid(np.arange(cols)-ccol, np.arange(rows)-crow)
    
    # Calculate the radius for each pixel in the Fourier transformed image
    r = np.sqrt(x**2 + y**2)
    
    # Create a mask of ones that has the same shape as the Fourier transformed image
    mask = np.ones_like(fshift)
    
    # Set the values in the mask to 0 where the radius is outside of the specified range
    mask[np.where((r < center_radius) | (r > outer_radius))] = 0
    
    # Multiply the Fourier transformed image by the mask
    fshift *= mask
    
    # Return the filtered Fourier transformed image
    return fshift