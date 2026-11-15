#!/bin/bash
# pull.sh <name> <trialname> <start> <end> : fetch frames [start,end) from box, encode to clips/<trialname>.mp4
N=$1; T=$2; S=$3; E=$4
D=/private/tmp/claude-501/c/fr/$T; mkdir -p $D /private/tmp/claude-501/c/clips
ssh aalto-sim-v00128 "cd /data/work/cine/out/$N/f; seq -f '%06g.jpg' $S $((E-1)) | tar cf - -T -" | tar xf - -C $D
ffmpeg -v error -y -framerate 25 -start_number $S -i $D/%06d.jpg -c:v libx264 -crf 17 -preset medium -pix_fmt yuv420p /private/tmp/claude-501/c/clips/$T.mp4 && rm -rf $D
ls -la /private/tmp/claude-501/c/clips/$T.mp4
