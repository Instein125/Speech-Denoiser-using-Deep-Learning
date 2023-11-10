import librosa
from preprocess import MinMaxNormalizer


class SoundGenerator:
    """SoundGenerator is responsible for generating audios from
    spectrograms.
    """
    def __init__(self, model, hop_length):
        self.model = model
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormalizer()

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms=self.model.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals
    
    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
        # reshape the log spectrogram
        spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2])

        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            
            # apply denormalisation
            denorm_log_spec = self._min_max_normaliser.denormalise(
                spectrogram, min_max_value["min"], min_max_value["max"])
            # log spectrogram -> spectrogram
            spec = librosa.db_to_amplitude(denorm_log_spec)
            # apply Griffin-Lim
            signal = librosa.istft(spec, hop_length=self.hop_length)
            # append signal to "signals"
            signals.append(signal)
        return signals