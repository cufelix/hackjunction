#!/bin/bash
# make_video.sh <frames_dir> <out.mp4> [max_seconds=40] : demo-camera jpgs (15 fps sim time) -> h264 mp4, <= max_seconds
# (a longer episode is sped up uniformly so the whole episode fits; speed factor is printed)
FF=/root/miniforge3/envs/isaac/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v9.0.2
D=$1; OUT=$2; MAX=${3:-40}
N=$(ls $D/*.jpg | wc -l); DUR=$(python3 -c "print($N/15.0)")
SP=$(python3 -c "print(max(1.0, $DUR/$MAX))")
echo "frames=$N dur=${DUR}s speedup=$SP"
$FF -y -loglevel error -framerate 15 -i $D/%06d.jpg -vf "setpts=PTS/$SP,fps=30,format=yuv420p" -c:v libx264 -crf 23 -preset medium -movflags +faststart -t $MAX $OUT
