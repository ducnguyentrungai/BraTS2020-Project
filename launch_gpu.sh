#!/bin/bash

srun --pty \
    --job-name=brats-seg \
    --partition=dgx-small  \
    --account=ddt_acc23 \
    --time=24:00:00 \
    /bin/bash
