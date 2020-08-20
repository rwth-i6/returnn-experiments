#!/usr/bin/env python3

import better_exchook
import os
import subprocess
import tempfile
from zipfile import ZipFile, ZipInfo
import contextlib

my_dir = os.path.dirname(os.path.abspath(__file__))


Parts = [
  "dev-clean", "dev-other",
  "test-clean", "test-other",
  "train-clean-100", "train-clean-360", "train-other-500"]


def generic_open(filename, mode="r"):
  """
  Wrapper around :func:`open`.
  Automatically wraps :func:`gzip.open` if filename ends with ``".gz"``.

  :param str filename:
  :param str mode: text mode by default
  :rtype: typing.TextIO|typing.BinaryIO
  """
  if filename.endswith(".gz"):
    import gzip
    if "b" not in mode:
      mode += "t"
    return gzip.open(filename, mode)
  return open(filename, mode)


def sh(*args):
  print("$ %s" % " ".join(args))
  subprocess.check_call(args)


@contextlib.contextmanager
def pushd(d):
  """
  :param str d: directory
  """
  assert os.path.isdir(d)
  old_working_dir = os.getcwd()
  os.chdir(d)
  yield
  os.chdir(old_working_dir)


def create_librispeech_txt(dataset_dir):
  """
  Create separate txt files to be used with :class:`returnn.OggZipDataset`.
  Example:
    https://github.com/rwth-i6/returnn-experiments/blob/master/2019-asr-e2e-trafo-vs-lstm/tedlium2/full-setup/03_convert_to_ogg.py

  :param str dataset_dir:
  """
  output_dir = dataset_dir
  with pushd(output_dir):
    for part in Parts:
      dest_meta_filename_gz = "%s.txt.gz" % part
      if os.path.exists(dest_meta_filename_gz):
        print("File exists:", dest_meta_filename_gz)
        continue
      dest_meta_filename = "%s.txt" % part
      dest_meta_file = open(dest_meta_filename, "w")
      dest_meta_file.write("[\n")
      zip_filename = "%s/%s.zip" % (dataset_dir, part)
      assert os.path.exists(zip_filename)
      zip_file = ZipFile(zip_filename)
      assert zip_file.filelist
      count_lines = 0
      for info in zip_file.filelist:
        assert isinstance(info, ZipInfo)
        path = info.filename.split("/")
        if path[0].startswith(part):
          subdir = path[0]  # e.g. "train-clean-100"
          assert subdir == part
          if path[-1].endswith(".trans.txt"):
            print("read", part, path[-1])
            for line in zip_file.read(info).decode("utf8").splitlines():
              seq_name, txt = line.split(" ", 1)  # seq_name is e.g. "19-198-0000"
              count_lines += 1
              ogg_filename = "%s/%s.flac.ogg" % ("/".join(path[:-1]), seq_name)
              ogg_bytes = zip_file.read(ogg_filename)
              assert len(ogg_bytes) > 0
              # ffprobe does not work correctly on piped input. That is why we have to store it in a temp file.
              with tempfile.NamedTemporaryFile(suffix=".ogg") as temp_file:
                temp_file.write(ogg_bytes)
                temp_file.flush()
                duration_str = subprocess.check_output(
                  ["ffprobe", temp_file.name,
                   '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'compact'],
                  stderr=subprocess.STDOUT).decode("utf8").strip()
                duration_str = duration_str.split("=")[-1]  # e.g. "format|duration=10.028000"
              assert float(duration_str) > 0  # just a check
              dest_meta_file.write(
                "{'text': %r, 'file': %r, 'seq_name': '%s', 'duration': %s},\n" % (
                  txt, ogg_filename, "%s-%s" % (part, seq_name), duration_str))
      assert count_lines > 0
      dest_meta_file.write("]\n")
      dest_meta_file.close()
      sh("gzip", dest_meta_filename)
      assert os.path.exists(dest_meta_filename_gz)


def extract_raw_strings_py(part):
  """
  :param str part:
  :rtype: str
  """
  dataset_dir = "%s/data/dataset-ogg" % my_dir
  dataset_path_prefix = "%s/%s" % (dataset_dir, part)
  py_txt_output_path = "%s/data/dataset/%s.py.txt.gz" % (my_dir, part)
  if os.path.exists(py_txt_output_path):
    print("File exists, skipping:", py_txt_output_path)
    return py_txt_output_path
  args = [
    "%s/returnn/tools/dump-dataset-raw-strings.py" % my_dir,
    "--dataset", repr({
      "class": "OggZipDataset",
      "path": ["%s.zip" % dataset_path_prefix, "%s.txt.gz" % dataset_path_prefix],
      'use_cache_manager': True,
      "audio": None,
      "targets": None,  # we will just use the raw strings
    }),
    "--out", py_txt_output_path]
  sh(*args)
  assert os.path.exists(py_txt_output_path)
  return py_txt_output_path


def main():
  os.makedirs("%s/data/dataset" % my_dir, exist_ok=True)

  create_librispeech_txt(dataset_dir="%s/data/dataset-ogg" % my_dir)

  trans_file = open("%s/data/dataset/train-trans-all.txt" % my_dir, "w")

  for part in Parts:
    py_txt_output_path = extract_raw_strings_py(part)

    if part.startswith("train"):
      py_txt = eval(generic_open(py_txt_output_path).read())
      assert isinstance(py_txt, dict) and len(py_txt) > 0
      example_key, example_value = next(iter(py_txt.items()))
      assert isinstance(example_key, str) and isinstance(example_value, str)
      for seq_tag, raw_txt in sorted(py_txt.items()):
        trans_file.write("%s\n" % raw_txt)

  trans_file.close()


if __name__ == '__main__':
    better_exchook.install()
    main()
