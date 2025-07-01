srun --pty \
    --job-name=brats \
    --partition=dgx-small  \
    --account=ddt_acc23 \
    --time=24:00:00 \
    /bin/bash
