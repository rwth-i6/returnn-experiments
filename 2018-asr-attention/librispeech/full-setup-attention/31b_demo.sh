#!/bin/bash

set -exv

# Make sure that you are running 31a_start_demo_server.sh in the background.
# Also, this uses sox and curl, which you should install before usage.

./returnn/demos/demo-record-and-push-to-webserver.py
