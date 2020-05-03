#!/bin/bash

for f in output/results/*
do
        echo
        echo $f
        cat $f
done
