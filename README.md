# manualCyborgPreprocessing
A collection of scripts and guides to complement fs2CyborgPreprocessing.

* Offline preprocessing: Count peaks in action potentials in 6 second 80 % overlapping sliding window of PSDs on noise reduced wave files. Description of procedure in offlineCyborgPreprocessing/README.md .
* Scripts to create noise tresholds used in fs2CyborgPreprocessing. Description of procedure in computeNoiseTresholdsForUseInfs2CyborgPreprocessing/README.md .

## Example: Offline preprocessing.

1. Select MEA2 Dopey experiment #2 (2017-03-20). Electrode 87 is converted to audio for visualization using convert_to_sound.py and displayed in Audacity.
![2 87 raw audio audacity](/img/2_87_raw_audio_audacity.png)

2. Use audacity's Noise reduction Effect to reduce noise in all electrodes, based on a selected noise segment in electrode 87. Export to wav files with "Export multiple".
![2 87 raw audio audacity](/img/2_87_raw_audio_audacity_selecting_noise_segment.png)
![2 87 raw audio audacity](/img/2_87_raw_audio_audacity_noise_reduced.PNG)

3. Use noise reduces wav files to make 6 seconds 80 % overlapping sliding windows producing histograms matrices of peak counts by running compute_frequency_ap_histograms.py.
Electrode 87 matrix along time axis.
![2 87 raw audio audacity](/img/2_87_AP_detection_from_sliding_6s_STFT_windows.PNG)
Electrode 87 matrix visualized.
![2 87 raw audio audacity](/img/2_87_AP_detection_from_sliding_6s_STFT_windows_3D.PNG)
Combined matrix of all electrodes is also produced.
![2 87 raw audio audacity](/img/2_all_electrodes_except_Ref_AP_detection_from_sliding_6s_STFT_windows_3D.PNG)

