"""Python reproduction of the EEG analyses in Shin et al. 2018, Sci. Data 5:180003.

Modules
-------
data     loading `Cleaned Data/` (.mat), markers, session boundaries, behaviour
filters  zero-phase Butterworth filtering (per recording session)
aar      port of the AAR toolbox ocular-artifact removal (iWASOBI + eog_fd)
bbci     ports of the BBCI toolbox functions used by the authors
ersp     port of EEGLAB newtimef (FFT/Hanning branch) for ERD/ERS maps
"""
