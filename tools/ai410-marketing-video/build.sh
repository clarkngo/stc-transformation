#!/bin/zsh
# AI 410 marketing video: turns the recorded scene frames into one MP4 and a poster.
# Usage: ./build.sh <frames-dir> <out.mp4>
set -euo pipefail
FRAMES=${1:?frames dir}; OUT=${2:?output mp4}
TMP=$(mktemp -d)
list=$TMP/list.txt; : > $list
for d in $FRAMES/*(/); do
  count=$(ls $d | wc -l | tr -d ' ')
  dur=$(echo "scale=3; $count/30" | bc)
  fo=$(echo "scale=3; $dur-0.35" | bc)
  ffmpeg -loglevel error -y -framerate 30 -i $d/f%05d.jpg \
    -vf "scale=1920:1080:flags=lanczos,fade=t=in:st=0:d=0.35,fade=t=out:st=$fo:d=0.35,format=yuv420p" \
    -c:v libx264 -preset slow -crf 20 -r 30 $TMP/${d:t}.mp4
  echo "file '$TMP/${d:t}.mp4'" >> $list
done
ffmpeg -loglevel error -y -f concat -safe 0 -i $list -c copy -movflags +faststart $OUT
ffmpeg -loglevel error -y -i $FRAMES/01-hook/f00001.jpg -vf scale=1280:720 -q:v 3 ${OUT:r}-poster.jpg
rm -rf $TMP
ffprobe -v error -show_entries format=duration,size -of default=nw=1 $OUT
