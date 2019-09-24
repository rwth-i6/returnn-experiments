#!/bin/bash

set -ex

# Convert tar.gz files to zip files, including ogg files.
# This is what we use in RETURNN dataset, because
# it allows better random access inside the file.

cd data
test -d dataset-ogg || mkdir dataset-ogg
zipdir="$(pwd)"/dataset-ogg
test -d $zipdir
tardir="$(pwd)"/dataset-raw
test -d $tardir
echo "tar dir: $tardir"
echo "OGG zip dir: $zipdir"
echo "TMPDIR = $TMPDIR"

# make sure that you have enough space on the tmp dir. should be at least 100 GB!
# maybe do sth like: export TMPDIR=/local/zeyer/tmp

function tar_to_zip() {
  name="$1"
  tarf="$tardir/$name.tar.gz"
  zipf="$name.zip"
  if test -e "$zipdir"/"$zipf"; then
    echo "zip file exists already: $zipf"
    return
  fi
  tmp=$(mktemp -d)
  pushd "$tmp"
  echo "zipping $zipf in temp dir $tmp"
  test -e "$tarf"  # should be absolute path
  tar xf "$tarf"
  find LibriSpeech -name "*.flac" -exec ffmpeg -i "{}" "{}.ogg" ";" -exec rm -f "{}" ";"
  zip -qdgds 500m -0 "$zipf" -r LibriSpeech
  ls -la "$zipf"
  mv "$zipf" "$zipdir"/
  popd
  rm -rf "$tmp"
}

tar_to_zip dev-clean
tar_to_zip dev-other
tar_to_zip test-clean
tar_to_zip test-other
tar_to_zip train-clean-100
tar_to_zip train-clean-360
tar_to_zip train-other-500

cd "$zipdir"
du -h *.zip
