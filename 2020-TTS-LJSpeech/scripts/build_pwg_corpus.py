import zipfile
import soundfile
import h5py
import numpy
import io
import sys
import os

FRAME_HOP=200

# create the zipfile and load the text file containing the sequence information
zip_dataset_path = sys.argv[1]
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

#segments = self._read_segment_list(tk.uncached_path(self.segment_file)) if self.segment_file else None
segments = None

feature_hdf = h5py.File(sys.argv[2], "r")
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
    matching_num_features = audio_len//FRAME_HOP
    features = features[:matching_num_features]

    audio_len = matching_num_features*FRAME_HOP
    audio = audio[:audio_len]

    path = os.path.join(sys.argv[3], name) + ".h5"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pwg_seq_h5_file = h5py.File(path)
    wavs = pwg_seq_h5_file.create_dataset("wave", (audio_len,), compression=None, dtype='float32')
    feats = pwg_seq_h5_file.create_dataset("feats", shape=features.shape, compression=None, dtype='float32')

    wavs[:] = audio
    feats[:] = numpy.asarray(features, dtype='float32')

    pwg_seq_h5_file.close()





