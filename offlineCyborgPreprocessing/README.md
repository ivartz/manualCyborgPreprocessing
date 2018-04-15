1. Get a raw recording of a MEA experiment in one ASCII csv file.
    It should be a file that comes directly from MultiChannel DataManager
    when converting an experiment
    (from MultiChannel Experimeter Recorder block) to ASCII form.

2. After adjusting input and output paths in convert_to_sound.py ,
    convert the csv file to separate wave files, one for each electrode
    by running
    python3 convert_to_sound.py

3. Import all sounds into Audacity and noise reduce all sounds
    using Audacity noise reduction plugin. Export all noise reduced sounds
    into a separate folder (still wave files).

4. After adjusting input and output paths in compute_frequency_ap_histograms ,
    convert the noise reduced sounds to preprocessed form by running
    python3 compute_frequency_ap_histograms.py
