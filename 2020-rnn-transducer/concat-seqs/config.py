
"""
Config for Sisyphus.
Here we declare which experiments to run.
"""

from sisyphus import *
import os


from recipe.dataset import common, bliss
from recipe import returnn, scoring


assert os.path.exists("%s/returnn" % tk.gs.BASE_DIR)
assert os.path.exists("%s/subword-nmt" % tk.gs.BASE_DIR)

# Note:
# /u/tuske/work/ASR/switchboard/corpus/xml/
#   has lower/upper case ("Dustin", "TV"), real numbers ("747").
# /work/asr3/irie/data/switchboard/corpora/
#   has all lower case ("dustin", "tv") (except special tokens like "[LAUGHTER]"),
#   and numbers converted to words ("seven hundred fourty").

bliss_files = {
  "train": Path("/u/tuske/work/ASR/switchboard/corpus/xml/train.corpus.gz"),
  "dev": Path("/u/tuske/work/ASR/switchboard/corpus/xml/dev.corpus.gz"),  # hub5'00
  "hub5e_01": Path("/u/tuske/work/ASR/switchboard/corpus/xml/hub5e_01.corpus.gz"),
  "rt03s": Path("/u/tuske/work/ASR/switchboard/corpus/xml/rt03s.corpus.gz")
}

bliss_files_irie = {
  "train": Path("/work/asr3/irie/data/switchboard/corpora/train.corpus.gz"),
  "dev": Path("/work/asr3/irie/data/switchboard/corpora/dev.corpus.gz"),  # hub5'00
  "hub5e_01": Path("/work/asr3/irie/data/switchboard/corpora/hub5e_01.corpus.gz"),
  "rt03s": Path("/work/asr3/irie/data/switchboard/corpora/rt03s.corpus.gz")
}

segment_files = {
  "train": Path("/u/zeyer/setups/switchboard/2017-12-11--returnn/dependencies/seg_train"),
  "cv": Path("/u/zeyer/setups/switchboard/2017-12-11--returnn/dependencies/seg_cv")
}

bliss.BlissToOggZipDatasetJob.ExistingCache.update({
  # These were created using /u/tuske corpora.
  (bliss_files["train"], segment_files["train"]): Path("/u/zeyer/setups/switchboard/dataset/data/train.zip"),
  (bliss_files["train"], segment_files["cv"]): Path("/u/zeyer/setups/switchboard/dataset/data/cv.zip"),
  (bliss_files["dev"], None): Path("/u/zeyer/setups/switchboard/dataset/data/dev.zip"),
  (bliss_files["hub5e_01"], None): Path("/u/zeyer/setups/switchboard/dataset/data/hub5e_01.zip"),
  (bliss_files["rt03s"], None): Path("/u/zeyer/setups/switchboard/dataset/data/rt03s.zip"),
})

datasets = {
  "train": bliss.bliss_to_ogg_zip("train", bliss_files["train"], segment_files["train"]),
  "cv": bliss.bliss_to_ogg_zip("cv", bliss_files["train"], segment_files["cv"]),
  "dev": bliss.bliss_to_ogg_zip("dev", bliss_files["dev"]),
  "hub5e_01": bliss.bliss_to_ogg_zip("hub5e_01", bliss_files["hub5e_01"]),
  "rt03s": bliss.bliss_to_ogg_zip("rt03s", bliss_files["rt03s"]),
}

train_trans_txt = bliss.bliss_to_txt(bliss_files["train"])
tk.register_output("swb-train-trans.txt.gz", train_trans_txt)

datasets_irie = {
  "train": bliss.bliss_to_ogg_zip("train_irie", bliss_files_irie["train"], segment_files["train"]),
  "cv": bliss.bliss_to_ogg_zip("cv_irie", bliss_files_irie["train"], segment_files["cv"]),
  "dev": bliss.bliss_to_ogg_zip("dev_irie", bliss_files_irie["dev"]),
  "hub5e_01": bliss.bliss_to_ogg_zip("hub5e_01_irie", bliss_files_irie["hub5e_01"]),
  "rt03s": bliss.bliss_to_ogg_zip("rt03s_irie", bliss_files_irie["rt03s"]),
}
datasets.update({"%s_irie" % key: value for (key, value) in datasets_irie.items()})

train_trans_irie_txt = bliss.bliss_to_txt(bliss_files_irie["train"])
tk.register_output("swb-train-trans-irie.txt.gz", train_trans_irie_txt)

text_post_process = [
  "get_replace('401K', 'four o one k')",
  "get_replace('401', 'four o one')",
  "english_cleaners_keep_special",
  "get_remove_chars(',/')"]

datasets_pp = {
  key: value.copy(name="%s-pp" % value.name, other_opts={"targets_post_process": text_post_process})
  for (key, value) in datasets.items()}
datasets.update({"%s_pp" % key: value for (key, value) in datasets_pp.items()})

train_trans_pp_txt = common.ogg_zip_dataset_to_txt(datasets["train_pp"])
tk.register_output("swb-train-trans-pp.txt.gz", train_trans_pp_txt)

common.CreateBpeJob.register_existing(
  txt_file=train_trans_txt,
  bpe_size=10000,
  output=common.BpeCodesVocab(
    codes="/u/zeyer/setups/switchboard/subwords/swb-bpe-codes",
    vocab="/u/zeyer/setups/switchboard/subwords/swb-vocab",
    unk="UNK"))

# Note: BPE1k by Irie was created using /work/asr3/irie (actually maybe using full Fisher?),
# different text preprocessing.
# WARNING: This must be used together with corpora using the same consistent text preprocessing.
bpe1k_irie = common.BpeCodesVocab(
   codes="/u/zeyer/setups/switchboard/subwords/irie-swbd_clean.bpe_code_1k",
   vocab="/u/zeyer/setups/switchboard/subwords/irie-vocab.swbd_clean.bpe1k",
   unk="UNK")

scoring.ScliteHubScoreJob.RefsStmFiles.update({
  "hub5e_00": Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"),
  "hub5e_01": Path("/u/tuske/bin/switchboard/hub5e_01.2.stm"),
  "rt03s": Path("/u/tuske/bin/switchboard/rt03s_ctsonly.stm"),
})
scoring.ScliteHubScoreJob.GlmFile = "/u/zeyer/setups/switchboard/dataset/data/kaldi-eval2000.glm"


# Overwrite experiments.score_hyps by this.
# noinspection PyUnusedLocal,PyShadowingNames
def score_hyps(experiment, dataset, hyps):
  """
  :param returnn.experiments.ExperimentFromConfig|None experiment: (unused)
  :param str dataset: from experiments.dataset_inference_keys
  :param Path hyps:
  :rtype: list[Path]
  """
  return scoring.ScliteHubScoreJob.create_by_corpus_name(
    name=dataset, hyps=hyps).output_results_txts_list


experiments = returnn.MultipleExperimentsFromConfigs(config_dir="config-train")
experiments.dataset_name = "swb"
experiments.dataset_inference_keys = ["dev", "hub5e_01", "rt03s"]
experiments.dataset_default_dev_sets = ["hub5e_00", "hub5e_00: Callhome"]
experiments.score_hyps = score_hyps
experiments.audio_norm_dataset = datasets["train"]
# The idea is that all used target-types and audio-feature-types need to be registered,
# and then are used by name from the RETURNN config.
for name, dataset in datasets.items():
  experiments.register_dataset(name, dataset)
experiments.register_audio("mfcc", common.ExtractAudioFeaturesOptions())
experiments.register_audio(
  "mfcc.norm-per-seq", common.ExtractAudioFeaturesOptions(norm_mean="per_seq", norm_std_dev="per_seq"))
experiments.register_audio(
  "logmel80", common.ExtractAudioFeaturesOptions(features="log_mel_filterbank", num_feature_filters=80))
experiments.register_audio(
  "logmel80.norm-per-seq",
  common.ExtractAudioFeaturesOptions(
    features="log_mel_filterbank", num_feature_filters=80,
    norm_mean="per_seq", norm_std_dev="per_seq"))
experiments.register_audio(
  "logmel80.no-norm",
  common.ExtractAudioFeaturesOptions(
    features="log_mel_filterbank", num_feature_filters=80, norm_mean=None, norm_std_dev=None))
experiments.register_audio(
  "logmel40.no-norm",
  common.ExtractAudioFeaturesOptions(
    features="log_mel_filterbank", num_feature_filters=40, norm_mean=None, norm_std_dev=None))
experiments.register_targets("bpe10k", common.txt_to_bpe(txt_file=train_trans_txt, bpe_size=10000))
experiments.register_targets("bpe1k", common.txt_to_bpe(txt_file=train_trans_txt, bpe_size=1000))
experiments.register_targets("bpe1k_pp", common.txt_to_bpe(txt_file=train_trans_pp_txt, bpe_size=1000))
experiments.register_targets("bpe1k_irie", bpe1k_irie)
experiments.load_experiments_from_config_dir()


from recipe.concat_seqs import ConcatSwitchboard
from recipe.returnn.experiments import SearchScoreResults
from recipe.scoring import ScliteHubScoreJob
# Note: Concat1 can still be different from the baseline because the dataset seq ordering is slightly different...
concat_nums = [1, 2, 4, 6, 8, 10, 20, 30, 50, 60, 80, 100]
for concat_num in concat_nums:
  ConcatSwitchboard.create_all_for_num(num=concat_num, register_output_prefix="concat_", experiments=experiments)

# We could also add Andres experiments, via separate base dir...
external_base_dir = "/u/zeyer/setups/switchboard/2019-10-22--e2e-bpe1k"
external_exps = [
  "base2.conv2l.specaug4a.ctc.devtrain",
  "base2.conv2l.specaug4a.ctc.devtrain.retrain1",
  "base2.conv2l.specaug4a.ctc.devtrain.retrain1.keeplast20",
  "rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03"
  ".pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain",
  "rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03"
  ".pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain.retrain1",
  "rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03"
  ".pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain.retrain1"
  ".keeplast20",
  "rna3c-base2_150-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03"
  ".mlr50.emit2.fl2.rep.fixmask.ctcalignfixsame.chunk60.encctc.devtrain",
  "rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03"
  ".pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.encctc.devtrain",
]

for external_exp in external_exps:
  external_dir = Path("%s/data-train/%s" % (external_base_dir, external_exp))
  exp = experiments.add_external_experiment(
    name="extern.%s" % external_exp,
    config_path=Path("%s/config-train/%s.config" % (external_base_dir, external_exp)),
    extern_dir=external_dir,
    extra_config_paths=[Path("%s/returnn_extern_config.py" % tk.gs.BASE_DIR)],
    external_dir=external_dir)

  results = SearchScoreResults(model=exp.best_model)
  tk.register_output("%s.txt" % (exp.get_name_for_hash(),), results.summarize_job.output_simple_txt)

  if "keeplast" in external_exp:
    continue

  # for beam_size in [1, 2, 4, 8, 12, 24, 32, 64, 128]:
  #   results = SearchScoreResults(
  #     model=exp.best_model, beam_size=beam_size,
  #     beam_search_rqmt={"time": 0.3 if beam_size <= 12 else 1.0})
  #   tk.register_output(
  #     "%s.beam%i.txt" % (exp.get_name_for_hash(), beam_size), results.summarize_job.output_simple_txt)
  #   if external_exp.startswith("rna"):
  #     results = SearchScoreResults(
  #       model=exp.best_model, beam_size=beam_size, beam_search_other_opts={"use_filtered_score": False},
  #       beam_search_rqmt={"time": 0.3 if beam_size <= 12 else 1.0})
  #     tk.register_output(
  #       "%s.beam%i.no-filter.txt" % (exp.get_name_for_hash(), beam_size), results.summarize_job.output_simple_txt)

  for concat_num in concat_nums:
    beam_sizes = [None]
    if external_exp.endswith(".retrain1"):  # multi beam sizes, keep a bit restricted...
      beam_sizes = [None, 1, 64]
    for beam_size in beam_sizes:
      postfix = "concat%i" % concat_num
      if beam_size:
        postfix += ".beam%i" % beam_size
      results = SearchScoreResults(
        model=exp.best_model, beam_size=beam_size,
        datasets=["%s_concat%i" % (name, concat_num) for name in ScliteHubScoreJob.OrigCorpusNames],
        beam_search_rqmt={"time": 0.3 if (concat_num <= 4 and not beam_size) else 1.0})
      tk.register_output(
        "%s.%s.txt" % (exp.get_name_for_hash(), postfix), results.summarize_job.output_simple_txt)
      if external_exp.startswith("rna") and False:
        results = SearchScoreResults(
          model=exp.best_model, beam_size=beam_size,
          datasets=["%s_concat%i" % (name, concat_num) for name in ScliteHubScoreJob.OrigCorpusNames],
          beam_search_other_opts={"use_filtered_score": False},
          beam_search_rqmt={"time": 0.3 if (concat_num <= 4 and not beam_size) else 1.0})
        tk.register_output(
          "%s.%s.no-filter.txt" % (exp.get_name_for_hash(), postfix), results.summarize_job.output_simple_txt)

from recipe.returnn.model import AverageModelJob
exp = experiments.experiments["extern.base2.conv2l.specaug4a.ctc.devtrain.retrain1.keeplast20"]
for n in [1, 3, 5, 10, 20]:
  models = [exp.get_train_model(exp.num_epochs - i) for i in reversed(range(n))]
  avg_model = AverageModelJob(models=models).output_model
  results = SearchScoreResults(model=avg_model)
  tk.register_output("%s.avglast%i.txt" % (exp.get_name_for_hash(), n), results.summarize_job.output_simple_txt)
models = [exp.get_train_model(exp.num_epochs - i * 6) for i in reversed(range(3))]
avg_model = AverageModelJob(models=models).output_model
results = SearchScoreResults(model=avg_model)
tk.register_output("%s.avglast3m.txt" % (exp.get_name_for_hash(),), results.summarize_job.output_simple_txt)

exp = experiments.experiments[
  "extern."
  "rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03"
  ".pretrain_less2_rep6.mlr50.emit2.fl2.rep.fixmask.ctcalignfix-ctcalign-p0-6l.chunk60.encctc.devtrain.retrain1"
  ".keeplast20"]
for n in [1, 5, 10, 20]:
  models = [exp.get_train_model(exp.num_epochs - i) for i in reversed(range(n))]
  avg_model = AverageModelJob(models=models).output_model
  results = SearchScoreResults(model=avg_model)
  tk.register_output("%s.avglast%i.txt" % (exp.get_name_for_hash(), n), results.summarize_job.output_simple_txt)
