import os
from easydict import EasyDict as edict
import yaml
from config import get_config
import utils
import joblib
import torch
from torch.nn.utils.rnn import pad_sequence
from utils import collect_files
import numpy as np
import espnet
import espnet.transform as trans
import espnet.transform.spec_augment as SPEC
import matplotlib.pyplot as plt
import random
import librosa
# from sox.transform import Transformer
import sox
import soundfile as sf
import copy
import string
import opensmile
from tqdm import tqdm

class DiCOVA(object):
    """Solver"""

    def __init__(self, config):
        self.config = config
        self.root = self.config.directories.dicova_root
        self.metadata = self.get_metadata()
        self.partition = self.get_partition()
        self.featdir = self.config.directories.dicova_logspect_feats

    def get_partition(self):
        if not os.path.exists('dicova_partitions.pkl'):
            files = collect_files(os.path.join(self.config.directories.dicova_root, 'LISTS'))
            folds = {}
            for file in files:
                name = file.split('/')[-1][:-4]
                pieces = name.split('_')
                train_val = pieces[0]
                fold = pieces[2]
                if fold in folds:
                    folds[fold][train_val] = file
                else:
                    folds[fold] = {train_val: file}
            fold_files = {}
            for fold, partition in folds.items():
                train = partition['train']
                with open(train) as f:
                    train_files = f.readlines()
                train_files = [x.strip() + '.wav' for x in train_files]
                """Get train positives and train negatives"""
                train_pos = []
                train_neg = []
                for file in train_files:
                    meta = self.get_file_metadata(file)
                    if meta['Covid_status'] == 'p':
                        train_pos.append(file)
                    elif meta['Covid_status'] == 'n':
                        train_neg.append(file)
                val = partition['val']
                with open(val) as f:
                    val_files = f.readlines()
                val_files = [x.strip() + '.wav' for x in val_files]
                val_pos = []
                val_neg = []
                for file in val_files:
                    meta = self.get_file_metadata(file)
                    if meta['Covid_status'] == 'p':
                        val_pos.append(file)
                    elif meta['Covid_status'] == 'n':
                        val_neg.append(file)
                fold_files[fold] = {'train_pos': train_pos, 'train_neg': train_neg,
                                    'val_pos': val_pos, 'val_neg': val_neg}
            joblib.dump(fold_files, 'dicova_partitions.pkl')
        else:
            fold_files = joblib.load('dicova_partitions.pkl')
        return fold_files

    def feature_path_partition(self):
        new_dict = {}
        for fold, fold_dictionary in self.partition.items():
            new_dict[fold] = {}
            for type_key, type_list in fold_dictionary.items():
                new_list = [os.path.join(self.featdir, x[:-4] + '.pkl') for x in type_list]
                new_dict[fold][type_key] = new_list
        self.feat_partition = new_dict

    def get_metadata(self):
        if not os.path.exists('dicova_metadata.pkl'):
            import csv
            lines = []
            with open(os.path.join(self.config.directories.dicova_root, 'metadata.csv'), newline='') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i > 0:
                        lines.append(row)
            metadata = {}
            for line in lines:
                metadata[line[0]] = {'Covid_status': line[1], 'Gender': line[2], 'Nationality': line[3]}
            joblib.dump(metadata, 'dicova_metadata.pkl')
        else:
            metadata = joblib.load('dicova_metadata.pkl')
        return metadata

    def get_file_metadata(self, file):
        filename = file.split('/')[-1][:-4]
        return self.metadata[filename]

    def get_augmented_file_metadata(self, file):
        filename = file.split('/')[-1][:-4]
        filename_chunks = filename.split('_')
        filename = filename_chunks[0] + '_' + filename_chunks[1]
        return self.metadata[filename]

    def get_features(self):
        """Add full filepaths"""
        partition = self.partition['1']
        filelist = partition['train_pos'] + partition['train_neg'] + partition['val_pos'] + partition['val_neg']
        new_filelist = []
        for file in filelist:
            new_filelist.append(os.path.join(self.root, 'AUDIO', file[:-4] + '.flac'))
        dump_dir = self.config.directories.dicova_wavs
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)
        utils.flac2wav(filelist=new_filelist, dump_dir=dump_dir)
        new_filelist = utils.collect_files(self.config.directories.dicova_wavs)
        utils.get_features(filelist=new_filelist, dump_dir=self.featdir)

    def get_opensmile_feats(self):
        # dump_dir = self.config.directories.dicova_wavs
        # utils.flac2wav(filelist=['/home/john/Documents/School/Spring_2021/DiCOVA_Train_Val_Data_Release_updated/AUDIO/YsVTYCAs_cough.flac'], dump_dir=dump_dir)

        files = utils.collect_files(self.config.directories.dicova_wavs)
        dump_root = self.config.directories.dicova_opensmile_feats
        if not os.path.isdir(self.config.directories.dicova_opensmile_feats):
            os.mkdir(self.config.directories.dicova_opensmile_feats)
        self.opensmile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        for file in tqdm(files):
            output = self.opensmile.process_file(file)
            opensmile_feats = np.squeeze(output.to_numpy())
            filename = file.split('/')[-1][:-4]
            dump_path = os.path.join(dump_root, filename + '.pkl')
            joblib.dump(opensmile_feats, dump_path)

    def get_test_files_and_feats(self):
        root = self.config.directories.dicova_test_root
        audio_path = os.path.join(root, 'AUDIO')
        files = utils.collect_files(audio_path)
        dump_dir = self.config.directories.dicova_test_wavs
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)
        utils.flac2wav(filelist=files, dump_dir=dump_dir)
        new_filelist = utils.collect_files(self.config.directories.dicova_test_wavs)
        dump_dir =self.config.directories.dicova_test_logspect_feats
        utils.get_features(filelist=new_filelist, dump_dir=dump_dir)

class Dataset(object):
    """Solver"""

    def __init__(self, config, params):
        """Get the data and supporting files"""
        self.config = config
        self.feature_dir = self.config.directories.freesound_logspect_feats
        'Initialization'
        self.list_IDs = params['files']
        self.mode = params["mode"]
        self.data_object = params['data_object']
        self.class2index, self.index2class = utils.get_class2index_and_index2class()
        self.incorrect_scaler = self.config.post_pretraining_classifier.incorrect_scaler
        self.specaugment = params['specaugment']
        self.specaug_probability = self.config.train.specaug_probability
        self.dataloader_temp_wavs = self.config.directories.dataloader_temp_wavs
        if not os.path.isdir(self.dataloader_temp_wavs):
            os.mkdir(self.dataloader_temp_wavs)
        self.opensmile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        if not os.path.exists('dicova_opensmile_maxvals.pkl'):
            utils.get_normalization_factors_opensmile()
        self.opensmile_norm_factors = joblib.load('dicova_opensmile_maxvals.pkl')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Get the data item'
        file = self.list_IDs[index]
        # """This may be a dumb way to do this but we want to change the filename here to go to .wav.
        #    It's coming from the .pkl files but I don't wanna change the pipeline so we fix it here."""
        # file = self.get_wav_from_pkl(file=file)
        if self.mode == 'train':
            metadata = self.data_object.get_augmented_file_metadata(file)
            label = self.class2index[metadata['Covid_status']]
            label = self.to_GPU(torch.from_numpy(np.asarray(label)))
        elif self.mode == 'val':
            metadata = self.data_object.get_file_metadata(file)
            label = self.class2index[metadata['Covid_status']]
            label = self.to_GPU(torch.from_numpy(np.asarray(label)))
        else:
            metadata = None
            label = None

        """We want to load the audio file. Then we want to perform the specaugment-esque transformations
           in the time domain. Then (1) compute spectrogram (2) compute OpenSMILE features."""

        if self.mode == 'train':
            feats = joblib.load(file)
            spectrogram = feats['spectrogram']
            opensmile_feats = feats['opensmile']
        elif self.mode == 'val':
            spectrogram = joblib.load(file)
            """Now we must also load the opensmile feats so change the filename"""
            filename = file.split('/')[-1][:-4]
            opensmile_path = os.path.join(self.config.directories.dicova_opensmile_feats, filename + '.pkl')
            opensmile_feats = joblib.load(opensmile_path)

        # audio, _ = librosa.core.load(file, sr=self.config.data.sr)
        # if self.specaugment and self.mode != 'test':
        #     new_audio = self.time_domain_spec_aug(audio=audio)
        # else:
        #     new_audio = copy.deepcopy(audio)

        # """Compute spectrogram"""
        # feature_processor = utils.Mel_log_spect()
        # spectrogram = feature_processor.get_Mel_log_spect(new_audio)
        #
        # """Get OpenSMILE features"""
        # # Generate random filename of length 8 to write to disk
        # temp_write_name = ''.join(random.choices(string.ascii_uppercase +
        #                              string.digits, k=8))
        # dump_path = os.path.join(self.dataloader_temp_wavs, temp_write_name + '.wav')
        # sf.write(dump_path, new_audio, self.config.data.sr, subtype='PCM_16')
        # output = self.opensmile.process_file(dump_path)
        # os.remove(dump_path)  # don't want to keep all those temporary files!
        # opensmile_feats = np.squeeze(output.to_numpy())

        """If we had the augmented dataset ahead of time, we could normalize each feature,
           but let's just normalize the feature vector. It should be similar."""
        opensmile_feats = opensmile_feats/self.opensmile_norm_factors

        spectrogram = self.to_GPU(torch.from_numpy(spectrogram))
        opensmile_feats = self.to_GPU(torch.from_numpy(opensmile_feats))
        spectrogram = spectrogram.to(torch.float32)
        opensmile_feats = opensmile_feats.to(torch.float32)

        # if self.specaugment:
        #     feats = joblib.load(file)
        #     x = random.uniform(0, 1)
        #     if x <= self.specaug_probability and self.mode != 'test':
        #         time_width = round(feats.shape[0]*0.1)
        #         aug_feats = SPEC.spec_augment(feats, resize_mode='PIL', max_time_warp=80,
        #                                                                max_freq_width=20, n_freq_mask=1,
        #                                                                max_time_width=time_width, n_time_mask=2,
        #                                                                inplace=False, replace_with_zero=True)
        #         # plt.subplot(211)
        #         # plt.imshow(feats.T)
        #         # plt.subplot(212)
        #         # plt.imshow(aug_feats.T)
        #         # plt.show()
        #         feats = self.to_GPU(torch.from_numpy(aug_feats))
        #     else:
        #         feats = self.to_GPU(torch.from_numpy(feats))
        # else:
        #     feats = self.to_GPU(torch.from_numpy(joblib.load(file)))
        """Get incorrect_scaler value"""
        if self.mode != 'test':
            if metadata['Covid_status'] == 'p':
                scaler = self.incorrect_scaler
            else:
                scaler = 1
            scaler = self.to_GPU(torch.from_numpy(np.asarray(scaler)))
            scaler = scaler.to(torch.float32)
            scaler.requires_grad = True
        else:
            scaler = None
        return file, spectrogram, opensmile_feats, label, scaler

    def to_GPU(self, tensor):
        if self.config.use_gpu == True:
            tensor = tensor.cuda()
            return tensor
        else:
            return tensor

    def get_wav_from_pkl(self, file):
        name = file.split('/')[-1][:-4]
        new_path = os.path.join(self.config.directories.dicova_wavs, name + '.wav')
        return new_path

    def get_fade_out(self, mask_length, fade_out_samples):
        start = np.linspace(start=1, stop=0, num=fade_out_samples)
        middle = np.zeros(shape=(mask_length-2*fade_out_samples))
        stop = np.linspace(start=0, stop=1, num=fade_out_samples)
        mask = np.concatenate((start, middle, stop))
        return mask

    def time_domain_spec_aug(self, audio, num_time_masks=2):
        num_audio_samples = len(audio)
        compressor = sox.Transformer()
        expander = sox.Transformer()
        """Step 1: time warping"""
        speed_up_factor = np.random.uniform(low=1.01, high=1.3)
        compressor.tempo(factor=speed_up_factor)
        slow_down_factor = np.random.uniform(low=0.7, high=0.99)
        expander.tempo(factor=slow_down_factor)
        # split audio into two pieces then choose randomly which piece to speed up or slow down
        split_point = int(num_audio_samples/2)
        first_half = audio[0:split_point]
        second_half = audio[split_point:]
        decision = np.random.uniform(low=0, high=1)
        if decision < 0.5:
            new_first_half = compressor.build_array(input_array=first_half, sample_rate_in=self.config.data.sr)
            new_second_half = expander.build_array(input_array=second_half, sample_rate_in=self.config.data.sr)
        else:
            new_first_half = expander.build_array(input_array=first_half, sample_rate_in=self.config.data.sr)
            new_second_half = compressor.build_array(input_array=second_half, sample_rate_in=self.config.data.sr)
        warped_audio = np.concatenate((new_first_half, new_second_half))
        num_audio_samples = len(warped_audio)
        # plt.subplot(211)
        # plt.plot(audio)
        # plt.subplot(212)
        # plt.plot(warped_audio)
        # plt.show()
        #
        # sf.write('warped_dummy.wav', warped_audio, self.config.data.sr, subtype='PCM_16')

        """Step 2: time masks. This equates to simple amplitude modulation."""
        min_time_mask_duration = 0.05*self.config.data.sr
        max_time_mask_duration = 0.55*self.config.data.sr
        fade_out_samples = 20
        if num_audio_samples > max_time_mask_duration + 10:
            for _ in range(num_time_masks):
                mask = np.ones_like(warped_audio)
                mask_start = np.random.randint(low=0, high=num_audio_samples-max_time_mask_duration)
                mask_dur = np.random.randint(low=min_time_mask_duration, high=max_time_mask_duration)
                mask_chunk = self.get_fade_out(mask_length=mask_dur, fade_out_samples=fade_out_samples)
                mask[mask_start:mask_start+mask_dur] = mask_chunk
                # old_warped = copy.deepcopy(warped_audio)
                warped_audio = warped_audio*mask
                # plt.subplot(311)
                # plt.plot(old_warped)
                # plt.subplot(312)
                # plt.plot(mask)
                # plt.subplot(313)
                # plt.plot(warped_audio)
                # plt.show()
        time_masked_audio = warped_audio
        """Step 3: Bandreject"""
        reject_frequency = np.random.uniform(low=80, high=int(self.config.data.sr/2)-300)
        width_q = np.random.uniform(low=0.05, high=0.1)
        bandreject = sox.Transformer()
        bandreject.bandreject(frequency=reject_frequency, width_q=width_q)
        bandrejected_audio = bandreject.build_array(input_array=time_masked_audio, sample_rate_in=self.config.data.sr)
        # sf.write('fully_processed_dummy.wav', bandrejected_audio, self.config.data.sr, subtype='PCM_16')
        # stop = None
        return bandrejected_audio

    def collate(self, data):
        files = [item[0] for item in data]
        spects = [item[1] for item in data]
        opensmile = [item[2] for item in data]
        labels = [item[3] for item in data]
        scalers = [item[4] for item in data]
        spects = pad_sequence(spects, batch_first=True, padding_value=0)
        opensmile = torch.stack([x for x in opensmile])
        if self.mode != 'test':
            labels = torch.stack([x for x in labels])
            scalers = torch.stack([x for x in scalers])
        return {'files': files, 'spects': spects, 'opensmile': opensmile,
                'labels': labels, 'scalers': scalers}


def main():
    config = get_config.get()
    dicova = DiCOVA(config=config)
    # dicova.get_features()
    # dicova.get_test_files_and_feats()
    dicova.get_opensmile_feats()

if __name__ == "__main__":
    main()