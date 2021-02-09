"""
Split .wav files with ffmpeg:
NAME=name.wav
ffmpeg -i $NAME.wav -f segment -segment_time 2 -c copy one_second/$NAME%03d.wav

https://petewarden.com/2017/07/17/a-quick-hack-to-align-single-word-audio-recordings/
"""
import glob
import os

file_names = glob.glob('background/*.wav')

for filename in file_names:
    print(filename)
    os.system('/tmp/extract_loudest_section/gen/bin/extract_loudest_section {} background2'.format(filename))
