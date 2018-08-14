#!/bin/bash

set -ex

# Convert tar.gz files to zip files.
# This is what we use in RETURNN dataset, because
# it allows better random access inside the file.

cd data
test -d dataset || mkdir dataset
zipdir="$(pwd)"/dataset
tardir="$(pwd)"/dataset-raw
echo "tar dir: $tardir"
echo "zip dir: $zipdir"
echo "TMPDIR = $TMPDIR"

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

