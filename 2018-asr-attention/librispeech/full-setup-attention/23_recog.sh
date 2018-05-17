#!/bin/bash

set -exv

# TODO: Better way to select some epochs, based on dev scores...

./tools/search.py returnn 100
