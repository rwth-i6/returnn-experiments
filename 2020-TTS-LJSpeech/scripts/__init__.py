from sisyphus import *

import pprint
import os
import stat
import subprocess
import glob

from recipe.default_paths import PWG_EXE, PWG_ROOT

class PWGTrain(Job):
    """

    """

    def __init__(self, pwg_config, pwg_train_dataset, pwg_dev_dataset, pwg_exe=PWG_EXE, pwg_src_root=PWG_ROOT):
        """

        :param dict pwg_config:
        :param Path pwg_train_dataset:
        :param Path pwg_dev_dataset:
        :param Path|str pwg_exe:
        :param Path|str pwg_src_root:
        """
        self.pwg_config = pwg_config
        self.pwg_train_dataset = pwg_train_dataset
        self.pwg_dev_dataset = pwg_dev_dataset

        self.pwg_exe = pwg_exe
        self.pwg_src_root = pwg_src_root

        self.config_file = self.output_path("pwg_config.py")
        self.model_folder = self.output_path("model", directory=True)

        max_steps = pwg_config['train_max_steps']
        interval = pwg_config['save_interval_steps']
        self.models = {i: self.output_path("model/checkpoint-%isteps.pkl" % i) for i in range(interval, max_steps+interval, interval)}

        self.rqmt = {'cpu': 4, 'gpu': 1, 'time': 120, 'mem': 16}

    def tasks(self):
        yield Task('create_files', mini_task=True)
        yield Task('run', 'run', rqmt=self.rqmt)

    def path_available(self, path):
        # if job is finised the path is available
        res = super().path_available(path)
        if res:
            return res

        # maybe the file already exists
        res = os.path.exists(path.get_path())
        if res:
            return res

        return False

    def build_train_call(self, checkpoint=None):
        train_call = [tk.uncached_path(self.pwg_exe), '-m', 'parallel_wavegan.bin.train']
        train_call += ['--config', tk.uncached_path(self.config_file)]
        train_call += ['--train-dumpdir', tk.uncached_path(self.pwg_train_dataset)]
        train_call += ['--dev-dumpdir', tk.uncached_path(self.pwg_dev_dataset)]
        train_call += ['--outdir', tk.uncached_path(self.model_folder)]
        if checkpoint:
            train_call += ['--resume', str(checkpoint)]

        return train_call

    def create_files(self):
        config_lines = []
        unreadable_data = {}

        pp = pprint.PrettyPrinter(indent=2, width=150)
        for k, v in sorted(self.pwg_config.items()):
            if pprint.isreadable(v):
                config_lines.append("'%s': %s," % (k, pp.pformat(v)))
            else:
                unreadable_data[k] = v

        if len(unreadable_data) > 0:
            assert False, "unreadable data"

        with open(tk.uncached_path(self.config_file), 'wt', encoding='utf-8') as f:
            f.write("{\n")
            f.write('\n'.join(config_lines))
            f.write("}\n")
        with open("run.sh", "wt") as f:
            f.write("#!/bin/bash\n")
            f.write("export PYTHONPATH=%s\n" % self.pwg_src_root)
            train_call = self.build_train_call()
            f.write(' '.join(train_call) + "\n")

        os.chmod('run.sh', stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def run(self):

        checkpoints = glob.glob(os.path.join(tk.uncached_path(self.model_folder), "*.pkl"))
        if len(checkpoints) > 0:
            print("resume training")
            latest_checkpoint = max(checkpoints, key=os.path.getctime)

            with open("resume.sh", "wt") as f:
                f.write("#!/bin/bash\n")
                f.write("export PYTHONPATH=%s\n" % self.pwg_src_root)
                train_call = self.build_train_call(checkpoint=latest_checkpoint)
                f.write(' '.join(train_call) + "\n")

            os.chmod('resume.sh', stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

            subprocess.check_call(['./resume.sh'])
        else:
            subprocess.check_call(['./run.sh'])


class PWGBuildDataset(Job):
    """
    This Job converts a dataset in zip_format and an HDF output file generated into the appriopriate format for
    the PWG training. The frame hop needs to be specified to adjust for the possible length mismatch between
    features and audio.
    """

    def __init__(self, zip_dataset, returnn_hdf_mels, frame_hop, segment_file=None, compression=None):
        """

        :param Path zip_dataset:
        :param Path returnn_hdf_mels:
        :param int frame_hop:
        :param Path segment_file:
        :param str|None compression:
        """
        self.zip_dataset = zip_dataset
        self.returnn_hdf_mels = returnn_hdf_mels
        self.frame_hop = frame_hop
        self.segment_file = segment_file
        self.compression = compression

        assert compression in [None, 'gzip', 'lzf']

        self.out = self.output_path("pwg_dataset", directory=True)

    def tasks(self):
        yield Task('run', mini_task=True)

    def _read_segment_list(self, segment_file):
        """
        read a list of segment names in either plain text or gzip

        :param str segment_file:
        """
        if segment_file.endswith(".gz"):
            import gzip
            segment_file_handle = gzip.open(segment_file)
            return set([s.decode() for s in segment_file_handle.read().splitlines()])
        else:
            segment_file_handle = open(segment_file, "rt")
            return set(segment_file_handle.read().splitlines())

    def run(self):

        import zipfile
        import soundfile
        import h5py
        import numpy
        import io

        # create the zipfile and load the text file containing the sequence information
        zip_dataset_path = tk.uncached_path(self.zip_dataset)
        zip_data = zipfile.ZipFile(zip_dataset_path)

        zip_name, _ = os.path.splitext(os.path.basename(zip_dataset_path))
        data = eval(zip_data.read("%s.txt" % zip_name))  # type: list

        assert data and isinstance(data, list)
        first_entry = data[0]
        assert isinstance(first_entry, dict)
        assert isinstance(first_entry["text"], str)
        assert isinstance(first_entry["file"], str)
        # currently there is only support for datasets containing the seq_name field
        assert isinstance(first_entry["seq_name"], str)
        assert isinstance(first_entry["duration"], float)

        segments = self._read_segment_list(tk.uncached_path(self.segment_file)) if self.segment_file else None

        feature_hdf = h5py.File(tk.uncached_path(self.returnn_hdf_mels), "r")
        feature_data = feature_hdf['inputs']
        feature_seq_names = feature_hdf['seqTags']
        feature_seq_lengths = feature_hdf['seqLengths']

        # store the positions of the sequences in the data in a dictionary for reading only selected sequences
        seq_positions = {}
        offset = 0
        for name, length in zip(feature_seq_names, feature_seq_lengths):
            name = name.decode('UTF-8') if isinstance(name, bytes) else name
            seq_positions[name] = (offset, offset+length[0])
            offset += length[0]

        for entry in data:
            name = entry['seq_name']
            if segments and (name not in segments):
                continue

            if name not in seq_positions.keys():
                print("%s was not in the features, skipping" % name)
                continue

            raw_audio = io.BytesIO(zip_data.read("%s/%s" % (zip_name, entry['file'])))
            audio, samplerate = soundfile.read(raw_audio, dtype='float32')
            audio_len = audio.shape[0]

            start_pos = seq_positions[name][0]
            end_pos = seq_positions[name][1]
            features = feature_data[start_pos:end_pos]

            # perform a correction of the audio and the feature data, as the returnn pipeline does not
            # exactly produces matching features. This can happen because of frame-padding
            matching_num_features = audio_len//self.frame_hop
            features = features[:matching_num_features]

            audio_len = matching_num_features*self.frame_hop
            audio = audio[:audio_len]

            path = os.path.join(tk.uncached_path(self.out), name) + ".h5"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pwg_seq_h5_file = h5py.File(path)
            wavs = pwg_seq_h5_file.create_dataset("wave", (audio_len,), compression=self.compression, dtype='float32')
            feats = pwg_seq_h5_file.create_dataset("feats", shape=features.shape, compression=self.compression, dtype='float32')

            wavs[:] = audio
            feats[:] = numpy.asarray(features, dtype='float32')

            pwg_seq_h5_file.close()





