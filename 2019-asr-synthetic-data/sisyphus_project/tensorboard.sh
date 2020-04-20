#!/bin/bash

if echo exit | nc localhost 6006; then
    echo "tensorboard already in use, quitting"
    pkill -x tensorboard
    sleep 5
fi

if [[ -d $1 ]]; then
    tensorboard --logdir $1/output &
    tensorboard_pid=$!
    while ! echo exit | nc localhost 6006; do sleep 1; done
    xdg-open http://localhost:6006
else
    echo "$1 is not a valid path"
fi
