
from sisyphus import *
from .utils import generic_open
import typing
import os
import shutil
import subprocess


class CalculateWordErrorRateJob(Job):
  def __init__(self, refs, hyps):
    """
    :param Path refs: Python txt format, seq->txt, whole words
    :param Path hyps: Python txt format, seq->txt, whole words
    """
    self.refs = refs
    self.hyps = hyps
    self.output_wer = self.output_path("wer.txt")

  def run(self):
    import subprocess
    args = [
      "%s/returnn/tools/calculate-word-error-rate.py" % tk.gs.BASE_DIR,
      "--expect_full",
      "--hyps", self.hyps.get_path(),
      "--refs", self.refs.get_path(),
      "--out", self.output_wer.get_path()]
    print("$ %s" % " ".join(args))
    subprocess.check_call(args)
    wer = float(open(self.output_wer.get_path()).read())
    print("WER: %f %%" % wer)

  def tasks(self):
    yield Task('run', rqmt={'cpu': 1, 'mem': 1, 'time': 0.1}, mini_task=True)


class ScliteJob(Job):
  """
  Run sclite.
  """

  def __init__(self, name, refs, hyps):
    """
    :param str name: e.g. dataset name (test-clean or so), just for the output reporting, not used otherwise
    :param Path refs: Python txt format, seq->txt, whole words
    :param Path hyps: Python txt format, seq->txt, whole words
    """
    self.name = name
    self.refs = refs
    self.hyps = hyps
    self.output_sclite_dir = self.output_path("sclite-out", directory=True)

  @staticmethod
  def create_stm(name, source_filename, target_filename):
    """
    :param str name:
    :param str source_filename: Python txt format
    :param str target_filename:
    :return: target_filename
    :rtype: str
    """
    # Example STM (.../switchboard/hub5e_00.2.stm):
    """
    ;; Reference file for english , generated Wed Apr  5 09:29:10 EDT 2000
    ;; CATEGORY "0" "" ""
    ;; LABEL "O" "Overall" "The Complete Test Set"
    ;; CATEGORY "1" "Corpus" ""
    ;; LABEL "EN" "Callhome" "Callhome conversations"
    ;; LABEL "SW" "Switchboard" "Switchboard conversations"
    ;; CATEGORY "2" "Sex" ""
    ;; LABEL "F" "Female" "Female Caller"
    ;; LABEL "M" "Male" "Male Caller"
    ;; CATEGORY "3" "Corpus/Sex" ""
    ;; LABEL "EN-F" "Callhome Female" "Callhome Female Caller"
    ;; LABEL "EN-M" "Callhome Male" "Callhome Male Caller"
    ;; LABEL "SW-F" "Switchboard Female" "Switchboard Female Caller"
    ;; LABEL "SW-M" "Switchboard Male" "Switchboard Male Caller"
    en_4156a 1 en_4156_A 301.85 302.48 <O,en,F,en-F>  oh yeah 
    en_4156a 1 en_4156_A 304.71 306.72 <O,en,F,en-F>  well i am going to have mine in two more classes 
    en_4156a 1 en_4156_A 307.63 311.16 <O,en,F,en-F>  no i am not well then i have to take my exams my orals but 
    en_4156a 1 en_4156_A 313.34 315.37 <O,en,F,en-F>  that is kind of what i would like to do 
    en_4156a 1 en_4156_A 316.83 319.20 <O,en,F,en-F>  i might even want to go on and get my p h d 
    en_4156a 1 en_4156_A 321.55 322.16 <O,en,F,en-F>  it is just that 
    """
    py_txt = eval(generic_open(source_filename).read())
    assert isinstance(py_txt, dict) and len(py_txt) > 0
    example_key, example_value = next(iter(py_txt.items()))
    assert isinstance(example_key, str) and isinstance(example_value, str)
    with generic_open(target_filename, "w") as f:
      f.write(';; CATEGORY "0" "" ""\n')
      f.write(';; LABEL "O" "%s" ""\n' % name)
      start = 0.0  # we invent some dummy start offsets and durations
      for seq_tag, raw_txt in sorted(py_txt.items()):
        # Dummy word durations. Dummy speaker name.
        f.write("%s 1 rec %f %f <O>  %s\n" % (
          seq_tag, start + 0.01, start + 0.99, raw_txt))
        start += 1
    return target_filename

  @staticmethod
  def create_ctm(source_filename, target_filename):
    """
    :param str source_filename: Python txt format
    :param str target_filename:
    :return: target_filename
    :rtype: str
    """
    # Example CTM:
    """
    ;; <name> <track> <start> <duration> <word> <confidence> [<n-best>]
    ;; hub5_00/en_4156a/1 (301.850000-302.480000)
    en_4156a 1 301.850000 0.283500 oh 0.99
    en_4156a 1 302.133500 0.283500 yeah 0.99
    ;; hub5_00/en_4156a/2 (304.710000-306.720000)
    en_4156a 1 304.710000 0.201000 well 0.99
    """
    py_txt = eval(generic_open(source_filename).read())
    assert isinstance(py_txt, dict) and len(py_txt) > 0
    example_key, example_value = next(iter(py_txt.items()))
    assert isinstance(example_key, str) and isinstance(example_value, str)
    with generic_open(target_filename, "w") as f:
      f.write(";; <name> <track> <start> <duration> <word> <confidence> [<n-best>]\n")
      start = 0.0  # we invent some dummy start offsets and durations
      for seq_tag, raw_txt in sorted(py_txt.items()):
        f.write(";; %s (%f-%f)\n" % (seq_tag, start + 0.01, start + 0.99))
        if raw_txt:
          words = raw_txt.split()
          # Dummy word durations. Dummy recording name.
          word_duration = 0.9 / len(words)
          for i in range(len(words)):
            f.write("%s 1 %f %f %s\n" % (seq_tag, start + 0.01 + i * word_duration, word_duration, words[i]))
        start += 1
    return target_filename

  def run(self):
    stm_filename = self.create_stm(self.name, self.refs.get_path(), "refs.stm")
    ctm_filename = self.create_ctm(self.hyps.get_path(), "hyps.ctm")
    args = [
      "%s/SCTK/bin/sclite" % tk.gs.BASE_DIR,
      '-r', stm_filename, 'stm',
      '-h', ctm_filename, 'ctm',
      "-e", "utf-8",
      '-o', 'all', '-o', 'dtl', '-o', 'lur',
      '-n', 'sclite',
      '-O', self.output_sclite_dir.get_path()]
    print("$ %s" % " ".join(args))
    for sclite_stdout_line in subprocess.check_output(args).splitlines():
      # Strip output a bit...
      if not sclite_stdout_line.strip():
        continue
      if b"Performing alignments for file" in sclite_stdout_line:
        continue
      if b"Segments For Channel" in sclite_stdout_line:
        continue
      print(sclite_stdout_line.decode("utf8"))

    raise NotImplementedError  # TODO gzip results...

  def tasks(self):
    yield Task('run', rqmt={'cpu': 1, 'mem': 1, 'time': 0.1}, mini_task=True)


class ScliteHubScoreJob(Job):
  """
  Wraps the SCTK hubscr.pl script, which is used to calculate the WER for Switchboard, Hub 5'00, Hub 5'01, rts03.
  """
  CorpusNameMap = {"dev": "hub5e_00"}
  OrigCorpusNames = ["hub5e_00", "hub5e_01", "rt03s"]
  ResultsSubsets = {
    "hub5e_00": ["Callhome", "Switchboard", "Overall"],
    "hub5e_01": ["Switchboard", "Switchboard-2 Phase III", "Switchboard-Cell", "Overall"],
    "rt03s": ["Swbd", "Fisher", "Overall"]}

  # This must be set in advance.
  RefsStmFiles = {}  # type: typing.Dict[str,Path]  # name -> path (use "hub5e_00", not "dev")
  GlmFile = None  # type: typing.Optional[str]  # filename

  @classmethod
  def create_by_corpus_name(cls, name, hyps):
    """
    :param str name: "hub5e_00", "hub5e_01" or "rt03s"
    :param Path hyps: Python txt format, seq->txt, whole words
    """
    name = cls.CorpusNameMap.get(name, name)
    assert name in cls.OrigCorpusNames or name in cls.RefsStmFiles
    assert name in cls.RefsStmFiles, "make sure you fill this dict before usage"
    return cls(name=name, stm=cls.RefsStmFiles[name], hyps=hyps)

  def __init__(self, name, stm, hyps):
    """
    :param str name: "hub5e_00", "hub5e_01" or "rt03s"
    :param Path stm: reference file (STM format)
    :param Path hyps: Python txt format, seq->txt, whole words
    """
    assert self.GlmFile, "make sure you set this before usage"
    self.name = name
    self.hyps = hyps
    self._ref_stm = stm
    self._glm = self.GlmFile
    self.ResultsSubsets = self.ResultsSubsets  # copy to self such that it pickles it
    self.output_dir = self.output_path("sclite-out", directory=True)
    # hub5e_00: Callhome, Switchboard, Overall
    # hub5e_01: Switchboard, Switchboard-2 Phase III, Switchboard-Cell, Overall
    # rt03s: Swbd, Fisher, Overall
    self.output_results_txt = self.output_path("results.txt")  # dict subset (str) -> WER% (float)
    self.output_results_txts = {
      subset: self.output_path("result-%s.txt" % subset.replace(" ", "-"))
      for subset in self.ResultsSubsets[name]}
    self.output_results_txts_list = [self.output_results_txts[subset] for subset in self.ResultsSubsets[name]]
    self.output_wer = self.output_path("wer.txt")  # overall WER%

  @staticmethod
  def create_ctm(name, ref_stm_filename, source_filename, target_filename):
    """
    :param str name: e.g. "hub5_00"
    :param str ref_stm_filename:
    :param str source_filename: Python txt format
    :param str target_filename: ctm file
    :return: target_filename
    :rtype: str
    """
    corpus_name_map = {"hub5e_00": "hub5_00", "hub5e_01": "hub5_01"}
    # Example CTM:
    """
    ;; <name> <track> <start> <duration> <word> <confidence> [<n-best>]
    ;; hub5_00/en_4156a/1 (301.850000-302.480000)
    en_4156a 1 301.850000 0.283500 oh 0.99
    en_4156a 1 302.133500 0.283500 yeah 0.99
    ;; hub5_00/en_4156a/2 (304.710000-306.720000)
    en_4156a 1 304.710000 0.201000 well 0.99
    """
    import re
    import decimal
    from decimal import Decimal

    py_txt = eval(generic_open(source_filename).read())
    assert isinstance(py_txt, dict) and len(py_txt) > 0
    example_key, example_value = next(iter(py_txt.items()))
    assert isinstance(example_key, str) and isinstance(example_value, str)

    # Read (ref) STM file, and write out (hyp) CTM file at the same time.
    with generic_open(target_filename, "w") as f:
      f.write(";; <name> <track> <start> <duration> <word> <confidence> [<n-best>]\n")

      seq_idx_in_tag = None
      last_tag = None
      last_end = None
      first_seq = True
      have_extended = False
      extended_seq_tag = None

      for line in generic_open(ref_stm_filename).read().splitlines():
        line = line.strip()
        if not line:
          continue
        # Example extended STM entry (added by ourselves):
        # _full_seq_tag "..."
        if line.startswith(";; _full_seq_tag "):
          if first_seq:
            have_extended = True
          else:
            assert have_extended
          assert not extended_seq_tag  # should have used (and reset) this
          m = re.match("^;; _full_seq_tag \"(.*)\"$", line)
          assert m, "unexpected line: %r" % line
          extended_seq_tag, = m.groups()
          continue
        if line.startswith(";;"):  # comments, or other meta info
          continue
        # Example STM entry (one seq):
        # en_4156a 1 en_4156_A 301.85 302.48 <O,en,F,en-F>  oh yeah
        m = re.match(
          "^([a-zA-Z0-9_]+)\\s+1\\s+([a-zA-Z0-9_]+)\\s+([0-9.]+)\\s+([0-9.]+)\\s+<([a-zA-Z0-9,\\-]+)>(.*)$", line)
        assert m, "unexpected line: %r" % line
        tag, tag2, start_s, end_s, flags, txt = m.groups()
        txt = txt.strip()
        first_seq = False
        if txt == "ignore_time_segment_in_scoring":
          continue
        if not txt:
          continue
        start = Decimal(start_s)
        end = Decimal(end_s)
        duration = end - start
        assert duration > 0.
        if tag != last_tag:
          seq_idx_in_tag = 1
          last_tag = tag
        else:
          assert start >= last_end - Decimal("0.01"), "line: %r" % line  # allow minimal overlap
          seq_idx_in_tag += 1
        last_end = end

        if extended_seq_tag:
          full_seq_tag = extended_seq_tag
          extended_seq_tag = None
        else:
          full_seq_tag = "%s/%s/%i" % (corpus_name_map.get(name, name), tag, seq_idx_in_tag)
        assert full_seq_tag in py_txt, "line: %r" % line
        hyp_raw_txt = py_txt.pop(full_seq_tag)
        words = hyp_raw_txt.split() if hyp_raw_txt else []
        # remove special symbols like [NOISE]
        words = [
          w for w in words
          if (w[:1] != "[" and w[-1:] != "]")]

        f.write(";; %s (%s-%s)\n" % (tag, start, end))
        f.write(";; full tag: %s\n" % full_seq_tag)
        f.write(";; ref: %s\n" % txt)
        if words:
          # Dummy word durations.
          word_duration = duration / len(words)
          for i in range(len(words)):
            f.write("%s 1 %.3f %.3f %s\n" % (
              tag, start + Decimal("0.01") + i * word_duration, word_duration * Decimal("0.9"), words[i]))

    assert not extended_seq_tag  # should have used (and reset) this

    if not have_extended:
      # There are some errors in the STM file. Just skip them.
      allowed_remaining = {"hub5_00/sw_4601a/28", "hub5_00/sw_4601a/29"}
      for tag in allowed_remaining:
        if tag in py_txt:
          py_txt.pop(tag)
    assert not py_txt
    return target_filename

  @staticmethod
  def parse_lur_file(filename):
    """
    :param str filename:
    :rtype: dict[str,float]
    """
    import re
    state = 0
    results = {}
    corpora = None
    p0, p1, p2 = None, None, None
    for line in generic_open(filename).read().splitlines():
      line = line.strip()
      if state == 0:
        if line[:2] == "|-":
          state = 1
      elif state == 1:
        assert line[:2] == "| "
        p = line.find("Corpus")
        assert p > 0
        p1, p2 = line.rfind("|", 0, p), line.find("|", p)
        assert 0 < p1 < p < p2
        assert line[p1 - 1:p1 + 1] == "||"
        p0 = line.rfind("|", 0, p1 - 1)
        assert 0 < p0 < p1
        state = 2
      elif state == 2:
        assert line[:2] == "|-"
        state = 3
      elif state == 3:
        assert line[p0 + 1:p1 - 1].strip() == "Overall"
        corpora = line[p1 + 1:p2].split("|")
        corpora = [c.strip() for c in corpora]
        state = 4
      elif state == 4:
        if line[:2] == "|=":
          state = 5
      elif state == 5:
        results_ = line[p1 + 1:p2].split("|")
        assert len(results_) == len(corpora)
        results_ = {key: value.strip() for (key, value) in zip(corpora, results_)}
        results_["Overall"] = line[p0 + 1:p1 - 1].strip()
        for key, value in list(results_.items()):
          # value is like "[76159] 24.8"
          m = re.match("^\\[([0-9]+)\\]\\s+([0-9.]+)$", value)
          assert m, "line %r, key %r, value %r" % (line, key, value)
          _, wer = m.groups()
          float(wer)  # test
          results[key] = float(wer)
        state = 6
    assert state == 6
    return results

  def run(self):
    stm_filename = "refs.stm"
    shutil.copy(self._ref_stm.get_path(), stm_filename)
    sclite_out_dir = self.output_dir.get_path()
    hyps_symlink_name = "%s/input_words.txt" % os.path.dirname(sclite_out_dir)
    if self.hyps.path.endswith(".gz"):
      hyps_symlink_name += ".gz"
    if not os.path.exists(hyps_symlink_name):
      os.symlink(self.hyps.get_path(), hyps_symlink_name)
    ctm_filename = self.create_ctm(
      name=self.name, ref_stm_filename=stm_filename,
      source_filename=self.hyps.get_path(), target_filename="%s/%s" % (sclite_out_dir, self.name))
    args = [
      "%s/SCTK/bin/hubscr.pl" % tk.gs.BASE_DIR,
      "-p", "%s/SCTK/bin" % tk.gs.BASE_DIR,
      "-V",
      "-l", "english",
      "-h", "hub5",
      "-g", self._glm,
      "-r", stm_filename,
      ctm_filename]
    print("$ %s" % " ".join(args))
    for sclite_stdout_line in subprocess.check_output(args).splitlines():
      # Strip output a bit...
      if not sclite_stdout_line.strip():
        continue
      if b"Performing alignments for file" in sclite_stdout_line:
        continue
      if b"Segments For Channel" in sclite_stdout_line:
        continue
      print(sclite_stdout_line.decode("utf8"))

    results = self.parse_lur_file("%s.filt.lur" % ctm_filename)
    print("Results:", results)
    with generic_open(self.output_results_txt.get_path(), "w") as f:
      f.write("%r\n" % (results,))
    with generic_open(self.output_wer.get_path(), "w") as f:
      f.write("%f\n" % results["Overall"])
    for subset in self.ResultsSubsets[self.name]:
      with generic_open(self.output_results_txts[subset].get_path(), "w") as f:
        dataset_name = "%s: %s" % (self.name, subset) if subset != "Overall" else self.name
        wer = results[subset]
        f.write("{'dataset': %r, 'keys': ['wer'], 'wer': %f}\n" % (dataset_name, wer))

    for fn in os.listdir(sclite_out_dir):
      self.sh("gzip %s/%s" % (sclite_out_dir, fn))

  def tasks(self):
    yield Task('run', rqmt={'cpu': 1, 'mem': 1, 'time': 0.1}, mini_task=True)
