import pandas as pd
from scipy.signal import resample
from tqdm import tqdm
from tqdm.auto import trange
import soundfile as sf
import json

from config import PATH, LIBRISPEECH_SAMPLING_RATE
from speakers import Data
from model.models import *
from utils import whiten

import torch


def predict_gender(audios, intervals, complex):

    step_seconds = 0.04

    model_path = 'model/weights/max_pooling__n_layers=7__n_filters=64__downsampling=1__n_seconds=3.torch'

    model_type = model_path.split('/')[-1].split('__')[0]
    model_name = model_path.split('/')[-1].split('.')[0]
    model_params = {i.split('=')[0]: float(i.split('=')[1]) for i in model_name.split('__')[1:]}

    # Here we assume that the model was trained on the LibriSpeech dataset
    model_sampling_rate = LIBRISPEECH_SAMPLING_RATE / model_params['downsampling']
    model_num_samples = int(model_params['n_seconds'] * model_sampling_rate)

    if model_type == 'max_pooling':
        model = ConvNet(int(model_params['n_filters']), int(model_params['n_layers']))
    elif model_type == 'dilated':
        model = DilatedNet(int(model_params['n_filters']), int(model_params['n_depth']),
                           int(model_params['n_stacks']))
    else:
        raise (ValueError, 'Model type not recognised.')

    model.load_state_dict(torch.load(model_path))
    model.double()
    model.cuda()
    model.eval()
    for i in trange(len(audios), desc="speakers"):
        speaker = audios[i].replace('.wav', '')

        ##############
        # Load audio #
        ##############
        audio_path = PATH + '/raw/voc/simple_audio/' + audios[i]
        audio, audio_sampling_rate = sf.read(audio_path)
        audio_duration_seconds = audio.shape[0] * 1. / audio_sampling_rate
        audio_duration_minutes = audio_duration_seconds / 60.

        step_samples = int(step_seconds * model_sampling_rate)
        step_samples_at_audio_rate = int(step_seconds * audio_sampling_rate)
        default_shape = None
        batch = []
        start_min = []
        pred = []
        mean_pitch = []
        max_pitch = []
        min_pitch = []
        num_zeros = []
        std_pitch = []
        pitch_measurements = []

        for j in trange(len(intervals[speaker]), desc="intervals", leave=False):
            start = float(intervals[speaker][j][0])
            end = float(intervals[speaker][j][1])
            start_samples = int(audio_sampling_rate * start)
            end_samples = int(audio_sampling_rate * end)
            step_samples = int(step_seconds * model_sampling_rate)
            step_samples_at_audio_rate = int(step_seconds * audio_sampling_rate)
            default_shape = None


            for lower in tqdm(range(start_samples, end_samples, step_samples_at_audio_rate), desc="predictions", leave=False):

                x = audio[lower:lower + (3 * audio_sampling_rate)]
                if x.shape[0] != 3 * audio_sampling_rate:
                    break

                sf.write(PATH + '/raw/clips/{}.wav'.format(speaker), x, audio_sampling_rate)
                sound = parselmouth.Sound(
                    PATH + '/raw/clips/{}.wav'.format(speaker))
                pitch = sound.to_pitch()
                pitch_values = pitch.selected_array['frequency']

                if pitch_values[pitch_values != 0].size != 0:
                    mean_pitch.append(np.mean(pitch_values[pitch_values != 0]))
                    std_pitch.append(np.std(pitch_values[pitch_values != 0]))
                    min_pitch.append(np.amin(pitch_values[pitch_values != 0]))
                    max_pitch.append(np.amax(pitch_values[pitch_values != 0]))
                    num_zeros.append(pitch_values[pitch_values == 0].size)
                    pitch_measurements.append(pitch_values[pitch_values != 0].size)
                    start_min.append(lower / 44100.)

                else:
                    mean_pitch.append(0)
                    std_pitch.append(0)
                    min_pitch.append(0)
                    max_pitch.append(0)
                    num_zeros.append(pitch_values[pitch_values == 0].size)
                    pitch_measurements.append(0)
                    start_min.append(lower / 44100.)

                os.remove(PATH + '/raw/clips/{}.wav'.format(speaker))

                x = torch.from_numpy(x).reshape(1, -1)

                x = whiten(x)

                # For me the bottleneck is this scipy resample call, increasing batch size doesn't make it any faster
                x = torch.from_numpy(
                    resample(x, model_num_samples, axis=1)
                ).reshape((1, 1, model_num_samples))

                y_hat = model(x).item()

                pred.append(y_hat)
                start_min.append(lower / 44100.)

        df = pd.DataFrame(data={'speaker': speaker, 'start_second': start_min, 'p': pred, 'mean_pitch': mean_pitch,
                                'max_pitch': max_pitch, 'min_pitch': min_pitch, 'num_zeros': num_zeros,
                                'std_pitch': std_pitch, 'pitch_measurements': pitch_measurements})

        df = df.assign(
            # Time in seconds of the end of the prediction fragment
            t_end=df['start_second'] + model_params['n_seconds'] / 60,
            # Time in seconds of the center of the prediction fragment
            t_center=df['start_second'] * 60 + model_params['n_seconds'] / 2.
        )
        df.to_csv(PATH + 'analyses/results/results_for_' + speaker + '.csv', index=False)


def get_audios():
    audios = []
    with open(PATH + '/speaker_list.txt', 'r') as filehandle:
        for line in filehandle:
            audios.append(line.replace('\n', ''))

    return audios


def main():
    print('Predicting {} GPU support'.format('with' if torch.cuda.is_available() else 'without'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    intervals = Data(False)
    speaker_to_intervals = intervals.speaker_to_intervals
    complex_transcripts = intervals.complex_transcripts
    print('creating and writing speakers_to_intervals.json')
    with open(PATH + '/speaker_to_intervals.json', 'w') as fp:
        json.dump(intervals.speaker_to_intervals, fp)
    with open(PATH + '/complex_transcripts.json', 'w') as fp:
        json.dump(intervals.complex_transcripts, fp)

    audio_list = get_audios()

    print(audio_list)

    predict_gender(audio_list, speaker_to_intervals, complex_transcripts)


if __name__ == "__main__":
    main()




