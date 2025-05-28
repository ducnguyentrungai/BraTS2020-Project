#!/bin/bash

srun --pty \
    --job-name=bratsJob \
    --partition=dgx-small\
    --account=ddt_acc23 \
    --time=6:00:00 \
    /bin/bash
