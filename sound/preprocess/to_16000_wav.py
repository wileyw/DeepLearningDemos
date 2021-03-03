


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

    for wav_path in glob.glob(os.path.join(INPUT_DIR, '*.wav')):
        name = os.path.split(wav_path)[1][:-len('.wav')]
        output_path = os.path.join(OUTPUT_DIR, name + '.wav')
        os.system('ffmpeg -i "{}" -ar 16000 "{}"'.format(wav_path, output_path))
        print(wav_path)
        print(output_path)

if __name__ == '__main__':
    main()
