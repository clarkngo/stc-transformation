#!/bin/zsh
# AI 410 marketing video: mixes background music under the finished voiceover.
# The music sits well below the voice, ducks further while the voice speaks, and fades at both ends.
# Usage: ./add-music.sh <voiceover.mp4> <music-file> <out.mp4> [music-gain-dB]
# Keep the music file out of git: its license doesn't allow sharing it on its own.
set -euo pipefail
VIDEO=${1:?voiceover mp4}; MUSIC=${2:?music file}; OUT=${3:?output mp4}; GAIN=${4:--21}
DUR=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 $VIDEO)
FADE=$(echo "$DUR-4" | bc)
ffmpeg -loglevel error -y -i $VIDEO -i $MUSIC -filter_complex "
[0:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[voice][key];
[1:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:$DUR,asetpts=N/SR/TB,volume=${GAIN}dB,afade=t=in:d=1.5,afade=t=out:st=$FADE:d=4[bed];
[bed][key]sidechaincompress=threshold=0.015:ratio=8:attack=40:release=700[ducked];
[voice][ducked]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.84:level=false[aud]" \
  -map 0:v -map "[aud]" -c:v copy -c:a aac -b:a 160k -ar 48000 -movflags +faststart $OUT
ffprobe -v error -show_entries format=duration,size -of default=nw=1 $OUT
