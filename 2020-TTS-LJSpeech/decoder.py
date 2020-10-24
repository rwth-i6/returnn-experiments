import argparse
from num2words import num2words
import numpy
import os
import subprocess
import sys
import torch
import yaml
import wave

sys.path.append("returnn")
sys.path.append("ParallelWaveGAN")

from parallel_wavegan import models as pwg_models
from parallel_wavegan.layers import PQMF
from returnn import rnn
from returnn.datasets.generating import StaticDataset, Vocabulary


def number_convert(word):
    try:
        f = float(word)
        return num2words(f)
    except:
        return word

def main():

    parser = argparse.ArgumentParser(description="TTS decoder running RETURNN TTS and an MB-MelGAN vocoder")
    parser.add_argument("--returnn_config", type=str, help="RETURNN config file (.config)")
    parser.add_argument("--vocab_file", type=str, help="RETURNN vocab file (.pkl)")
    parser.add_argument("--pronunciation_lexicon", type=str, help="CMU style pronuncation lexicon")
    parser.add_argument("--pwg_config", type=str, help="ParallelWaveGAN config (.yaml)")
    parser.add_argument("--pwg_checkpoint", type=str, help="ParallelWaveGAN checkpoint (.pkl)")

    args = parser.parse_args()

    # Initialize RETURNN
    rnn.init(args.returnn_config)
    rnn.engine.use_search_flag = True # enable search mode
    rnn.engine.init_network_from_config(rnn.config)

    returnn_vocab = Vocabulary(vocab_file=args.vocab_file, unknown_label=None)
    returnn_output_dict = {'output': rnn.engine.network.get_default_output_layer().output.placeholder}

    # Initialize PWG
    pwg_config = yaml.load(open(args.pwg_config), Loader=yaml.Loader)
    pyt_device = torch.device("cpu")
    generator = pwg_models.MelGANGenerator(**pwg_config['generator_params'])
    generator.load_state_dict(
        torch.load(args.pwg_checkpoint, map_location="cpu")["model"]["generator"])
    generator.remove_weight_norm()
    pwg_model = generator.eval().to(pyt_device)
    pwg_pad_fn = torch.nn.ReplicationPad1d(
        pwg_config["generator_params"].get("aux_context_window", 0))
    pwg_pqmf = PQMF(pwg_config["generator_params"]["out_channels"]).to(pyt_device)

    # load a CMU dict style pronunciation table
    pronunciation_dictionary = {}
    with open(args.pronunciation_lexicon, "rt") as lexicon:
        for lexicon_entry in lexicon.readlines():
            word, phonemes = lexicon_entry.strip().split(" ", maxsplit=1)
            pronunciation_dictionary[word] = phonemes.split(" ")
    
    # Tokenizer perl command
    tokenizer = ["perl", "./scripts/tokenizer/tokenizer.perl", "-l", "en", "-no-escape"]
    
    audios = []

    for line in sys.stdin.readlines():
        line = line.strip().lower()
        # run perl tokenizer as external script
        p = subprocess.Popen(tokenizer, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        line = p.communicate(input=line.encode("UTF-8"))[0].decode("UTF-8").strip()
        p.terminate()
        print(line)

        # apply num2wordsn and pronunciation dict
        words = list(map(number_convert, line.split(" ")))
        print(words)
        phoneme_sequence = " _ ".join([" ".join(pronunciation_dictionary[w]) for w in words if w in pronunciation_dictionary.keys()])
        phoneme_sequence += " _ ~"

        try:
            classes = numpy.asarray(returnn_vocab.get_seq(phoneme_sequence), dtype="int32")
            feed_dict = {'classes': classes}
            dataset = StaticDataset([feed_dict], output_dim={'classes': (77, 1)})
            result = rnn.engine.run_single(dataset, 0, returnn_output_dict)
        except Exception as e:
            print(e)
            raise e

        feature_data = numpy.squeeze(result['output']).T
        print(feature_data.shape)
        
        with torch.no_grad():
            input_features = pwg_pad_fn(torch.from_numpy(feature_data).unsqueeze(0)).to(pyt_device)
            audio_waveform = pwg_pqmf.synthesis(pwg_model(input_features)).view(-1).cpu().numpy()

        audios.append(numpy.asarray(audio_waveform*(2**15-1), dtype="int16").tobytes())

    for i, audio in enumerate(audios):
        wave_writer = wave.open("out_%i.wav" % i, "wb")
        wave_writer.setnchannels(1)  
        wave_writer.setframerate(16000)
        wave_writer.setsampwidth(2)
        wave_writer.writeframes(audio)
        wave_writer.close()


if __name__ == "__main__":
    main()
