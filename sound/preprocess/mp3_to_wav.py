


import os
import glob
import sys

def main():
    if len(sys.argv) < 3:
        print('Usage: python3 mp3_to_wav.py INPUT_DIR OUTPUT_DIR')
        return

    INPUT_DIR = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for mp3_path in glob.glob(os.path.join(INPUT_DIR, '*.mp3')):
        name = os.path.split(mp3_path)[1][:-len('.mp3')]
        output_path = os.path.join(OUTPUT_DIR, name + '.wav')
        os.system('ffmpeg -i "{}" -ar 16000 "{}"'.format(mp3_path, output_path))
        print(mp3_path)
        print(output_path)

if __name__ == '__main__':
    main()
