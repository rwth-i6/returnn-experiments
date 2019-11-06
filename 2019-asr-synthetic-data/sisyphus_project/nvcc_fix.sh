#!/bin/bash
set -ex

sed -i.bak 's/"cu",/"cu", "-DNDEBUG",/g' src/returnn/TFUtil.py