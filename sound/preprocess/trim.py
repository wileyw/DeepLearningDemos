"""
Convert to .wav
ffmpeg -i input.m4a output.wav

Split .wav files with ffmpeg:
NAME=name.wav
ffmpeg -i $NAME.wav -f segment -segment_time 2 -c copy one_second/$NAME%03d.wav

python3 to_16000_wav.py INPUT_DIR OUTPUT_DIR
python3 trim.py INPUT_DIR OUTPUT_DIR

https://petewarden.com/2017/07/17/a-quick-hack-to-align-single-word-audio-recordings/

NOTE: Run make from the extract_loudest_section repo before running this script
"""
import glob
import os
import sys

def main():
    if len(sys.argv) < 3:
        print('Usage: python3 trim.py INPUT_DIR OUTPUT_DIR')
        return

    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    file_names = glob.glob(os.path.join(sys.argv[1], '*.wav'))
    for filename in file_names:
        print(filename)
        os.system('/tmp/extract_loudest_section/gen/bin/extract_loudest_section "{}" "{}"'.format(filename, sys.argv[2]))

if __name__ == '__main__':
    main()
