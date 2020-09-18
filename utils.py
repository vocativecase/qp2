import torch
import os
import shutil
from config import PATH, BOX_PATH
import pandas as pd

def whiten(batch, rms=0.038021):
    """This function whitens a batch so each sample has 0 mean and the same root mean square amplitude i.e. volume."""
    # Subtract mean
    sample_wise_mean = batch.mean(dim=1)
    whitened_batch = batch-sample_wise_mean.repeat([batch.shape[1], 1]).transpose(dim0=1, dim1=0)

    # Divide through
    rescaling_factor = rms/ torch.sqrt(torch.mul(batch, batch).mean(dim=1))
    whitened_batch = whitened_batch*rescaling_factor.repeat([batch.shape[1], 1]).transpose(dim0=1, dim1=0)
    return whitened_batch


def evaluate(model, dataloader, preprocessor):
    """
    This function evaluates the performance of a model on a dataset. I will use this to determine when the model reaches
    peak generalisation performance and save the weights.
    :param model: Model to evaluate
    :param dataloader: An instance of a pytorch DataLoader class
    :param preprocessor: Function that takes a batch and performs any required preprocessing
    :return: Accuracy of the model on the data supplied by dataloader
    """
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            batch, labels = data

            batch = preprocessor(batch)

            predicted = model(batch)
            total += labels.size(0)
            correct += ((predicted > 0.5)[:, 0] == labels.cuda().byte()).cpu().sum().numpy()

    return correct * 1.0 / total


def clean_transcripts():
    for root, folders, files in os.walk(PATH + '/raw/voc/transcripts/'):
        for f in files:
            if f.endswith('.txt'):
                new_lines = []
                with open(PATH + '/raw/voc/transcripts/' + f, 'r') as fp:
                    for line in fp:
                        split = line.split('\t')
                        if split[4] != '\n':
                            new_lines.append(line)
                with open(PATH + '/raw/voc/transcripts/' + f, 'w+') as fp2:
                    fp2.writelines("%s" % line for line in new_lines)


def speakers_to_txt():
    speaker_list = []

    for root, folders, files in os.walk(PATH + '/raw/voc/simple_audio'):
        for f in files:
            if f.endswith('.wav'):
                speaker_list.append(f)

    with open('raw/voc/speaker_list.txt', 'w') as fp:
        fp.writelines("%s\n" % speaker for speaker in speaker_list)

def fetch_audio():
    local_transcripts = []
    box_transcripts = []
    box_audio = []
    local_audio = []

    data = Data(False)
    speakers = data.all_speakers


    for root, folders, files in os.walk(BOX_PATH):
        for f in files:
            if f.endswith('.wav'):
                box_audio.append(f.replace('.wav', ''))

    for root, folders, files in os.walk(PATH + '/raw/voc/audio/'):
        for f in files:
            if f.endswith('.wav'):
                local_audio

    for speaker in data.missing_audios:
        if speaker in box_audio:
            source = BOX_PATH + speaker + '.wav'
            destination = PATH + '/raw/voc/simple_audio/' + speaker + '.wav'
            print('Copying {}'.format(speaker))
            shutil.copyfile(source, destination)


