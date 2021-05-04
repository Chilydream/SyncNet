from scipy.io import wavfile


sample_rate, audio = wavfile.read('./data/00133.wav')
print(sample_rate)
print(audio.shape)