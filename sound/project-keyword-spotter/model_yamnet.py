# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keyword spotter model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import audio_recorder
import mel_features
import numpy as np
import queue
import os

if os.uname().sysname == 'Linux':
    import tflite_runtime.interpreter as tflite
else:
    import tensorflow.lite as tflite
import platform


# import matplotlib.pyplot as plt
# plt.ion()


EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]

q = queue.Queue()

logging.basicConfig(
    stream=sys.stdout,
    format="%(levelname)-8s %(asctime)-15s %(name)s %(message)s")
audio_recorder.logger.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def get_queue():
  return q


class Uint8LogMelFeatureExtractor(object):
  """Provide uint8 log mel spectrogram slices from an AudioRecorder object.

  This class provides one public method, get_next_spectrogram(), which gets
  a specified number of spectral slices from an AudioRecorder.
  """

  def __init__(self, num_frames_hop=48):
    self.spectrogram_window_length_seconds = 0.025
    self.spectrogram_hop_length_seconds = 0.010
    self.num_mel_bins = 64
    self.frame_length_spectra = 96
    if self.frame_length_spectra % num_frames_hop:
        raise ValueError('Invalid num_frames_hop value (%d), '
                         'must devide %d' % (num_frames_hop,
                                             self.frame_length_spectra))
    self.frame_hop_spectra = num_frames_hop
    self._norm_factor = 3
    self._clear_buffers()

  def _clear_buffers(self):
    self._audio_buffer = np.array([], dtype=np.int16).reshape(0, 1)
    self._spectrogram = np.zeros((self.frame_length_spectra, self.num_mel_bins),
                                 dtype=np.float32)

  def _spectrogram_underlap_samples(self, audio_sample_rate_hz):
    return int((self.spectrogram_window_length_seconds -
                self.spectrogram_hop_length_seconds) * audio_sample_rate_hz)

  def _frame_duration_seconds(self, num_spectra):
    return (self.spectrogram_window_length_seconds +
            (num_spectra - 1) * self.spectrogram_hop_length_seconds)
  def compute_spectrogram_and_normalize(self, audio_samples, audio_sample_rate_hz):
    spectrogram = self._compute_spectrogram(audio_samples, audio_sample_rate_hz)

    spectrogram -= np.mean(spectrogram, axis=0)
    if self._norm_factor:
      spectrogram /= self._norm_factor * np.std(spectrogram, axis=0)
      spectrogram += 1
      spectrogram *= 127.5
    return np.maximum(0, np.minimum(255, spectrogram)).astype(np.float32)

  def _compute_spectrogram(self, audio_samples, audio_sample_rate_hz):
    """Compute log-mel spectrogram and scale it to uint8."""
    samples = audio_samples.flatten() / float(2**15)
    spectrogram = 30 * (
        mel_features.log_mel_spectrogram(
            samples,
            audio_sample_rate_hz,
            log_offset=0.001,
            window_length_secs=self.spectrogram_window_length_seconds,
            hop_length_secs=self.spectrogram_hop_length_seconds,
            num_mel_bins=self.num_mel_bins,
            lower_edge_hertz=60,
            upper_edge_hertz=3800) - np.log(1e-3))
    return spectrogram

  def _get_next_spectra(self, recorder, num_spectra):
    """Returns the next spectrogram.

    Compute num_spectra spectrogram samples from an AudioRecorder.
    Blocks until num_spectra spectrogram slices are available.

    Args:
      recorder: an AudioRecorder object from which to get raw audio samples.
      num_spectra: the number of spectrogram slices to return.

    Returns:
      num_spectra spectrogram slices computed from the samples.
    """
    required_audio_duration_seconds = self._frame_duration_seconds(num_spectra)
    logger.info("required_audio_duration_seconds %f",
                required_audio_duration_seconds)
    required_num_samples = int(
        np.ceil(required_audio_duration_seconds *
                recorder.audio_sample_rate_hz))
    print(required_num_samples)
    logger.info("required_num_samples %d, %s", required_num_samples,
                str(self._audio_buffer.shape))
    # import ipdb; ipdb.set_trace()
    audio_samples = np.concatenate(
        (self._audio_buffer,
         recorder.get_audio(required_num_samples - len(self._audio_buffer))[0]))
    self._audio_buffer = audio_samples[
        required_num_samples -
        self._spectrogram_underlap_samples(recorder.audio_sample_rate_hz):]
    spectrogram = self._compute_spectrogram(
        audio_samples[:required_num_samples], recorder.audio_sample_rate_hz)
    assert len(spectrogram) == num_spectra
    return spectrogram

  def get_next_spectrogram(self, recorder):
    """Get the most recent spectrogram frame.

    Blocks until the frame is available.

    Args:
      recorder: an AudioRecorder instance which provides the audio samples.

    Returns:
      The next spectrogram frame as a uint8 numpy array.
    """
    assert recorder.is_active
    logger.info("self._spectrogram shape %s", str(self._spectrogram.shape))
    self._spectrogram[:-self.frame_hop_spectra] = (
        self._spectrogram[self.frame_hop_spectra:])
    self._spectrogram[-self.frame_hop_spectra:] = (
        self._get_next_spectra(recorder, self.frame_hop_spectra))
    # Return a copy of the internal state that's safe to persist and won't
    # change the next time we call this function.
    logger.info("self._spectrogram shape %s", str(self._spectrogram.shape))
    spectrogram = self._spectrogram.copy()
    spectrogram -= np.mean(spectrogram, axis=0)
    if self._norm_factor:
      spectrogram /= self._norm_factor * np.std(spectrogram, axis=0)
      spectrogram += 1
      spectrogram *= 127.5
    return np.maximum(0, np.minimum(255, spectrogram)) #.astype(np.uint8)
    # return spectrogram


def read_labels(filename):
  # The labels file can be made something like this.
  f = open(filename, "r")
  lines = f.readlines()
  #return ['negative'] + [l.rstrip() for l in lines]
  return [l.rstrip() for l in lines]


def read_commands(filename):
  # commands should consist of a label, a command and a confidence.
  f = open(filename, "r")
  commands = {}
  lines = f.readlines()
  for command, key, confidence in [l.rstrip().split(',') for l in lines]:
    commands[command] = { 'key': key, 'conf': 0.4}
    if confidence and 0 <= float(confidence) <= 1:
      commands[command]['conf'] = float(confidence)
  return commands


def get_output(interpreter):
    """Returns entire output, threshold is applied later."""
    return output_tensor(interpreter, 0)

def output_tensor(interpreter, i):
    """Returns dequantized output tensor if quantized before."""
    output_details = interpreter.get_output_details()[i]
    output_data = np.squeeze(interpreter.tensor(output_details['index'])())
    if 'quantization' not in output_details:
        return output_data
    scale, zero_point = output_details['quantization']
    if scale == 0:
        return output_data - zero_point
    return scale * (output_data - zero_point)


def input_tensor(interpreter):
    """Returns the input tensor view as numpy array."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]


def set_input(interpreter, data):
    """Copies data to input tensor."""
    interpreter_shape = interpreter.get_input_details()[0]['shape']
    input_tensor(interpreter)[:,:] = np.reshape(data, interpreter_shape[1:3])


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')

    if os.uname().sysname == 'Linux' and 'edgetpu' in model_file:
        return tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB,
                                {'device': device[0]} if device else {})
        ])
    else:
        # If sysname == 'Darwin'
        return tflite.Interpreter(model_path=model_file)


def add_model_flags(parser):
  parser.add_argument(
      "--model_file",
      help="File path of TFlite model.",
      default="models/model.tflite")
      # default="models/voice_commands_v0.7_edgetpu.tflite")
  parser.add_argument("--mic", default=None,
                      help="Optional: Input source microphone ID.")
  parser.add_argument(
      "--num_frames_hop",
      default=48,
      help="Optional: Number of frames to wait between model inference "
      "calls. Smaller numbers will reduce the latancy while increasing "
      "compute cost. Must devide 198. Defaults to 33.")
  parser.add_argument(
      "--sample_rate_hz",
      default=16000,
      help="Optional: Sample Rate. The model expects 16000. "
      "However you may alternative sampling rate that may or may not work."
      "If you specify 48000 it will be downsampled to 16000.")


def plot_spectrogram(input_data):
  # Convert to frequencies to log scale and transpose so that the time is
  # represented in the x-axis (columns).
  plt.imshow(np.squeeze(input_data).T)
  plt.draw()
  plt.pause(0.001)

def softmax(x, axis=None):
  x = x - x.max(axis=axis, keepdims=True)
  y = np.exp(x)
  return y / y.sum(axis=axis, keepdims=True)

def classify_audio(audio_device_index, interpreter, labels_file,
                   commands_file=None,
                   result_callback=None, dectection_callback=None,
                   sample_rate_hz=16000,
                   negative_threshold=0.6, num_frames_hop=33):
  """Acquire audio, preprocess, and classify."""
  # Initialize recorder.
  AUDIO_SAMPLE_RATE_HZ = sample_rate_hz
  downsample_factor = 1
  if AUDIO_SAMPLE_RATE_HZ == 48000:
    downsample_factor = 3
  # Most microphones support this
  # Because the model expects 16KHz audio, we downsample 3 fold
  recorder = audio_recorder.AudioRecorder(
      AUDIO_SAMPLE_RATE_HZ,
      downsample_factor=downsample_factor,
      device_index=audio_device_index)
  feature_extractor = Uint8LogMelFeatureExtractor(num_frames_hop=num_frames_hop)
  labels = read_labels(labels_file)
  if commands_file:
    commands = read_commands(commands_file)
  else:
    commands = {}
  logger.info("Loaded commands: %s", str(commands))
  logger.info("Recording")
  timed_out = False

  # Testing
  if False:
    sample_data = 'data/mini_speech_commands/down/e71b4ce6_nohash_1.wav'

    import tensorflow as tf
    import os
    def decode_audio(audio_binary):
      audio, _ = tf.audio.decode_wav(audio_binary)
      return tf.squeeze(audio, axis=-1)

    def get_label(file_path):
      parts = tf.strings.split(file_path, os.path.sep)

      # Note: You'll use indexing here instead of tuple unpacking to enable this 
      # to work in a TensorFlow graph.
      return parts[-2]

    def get_waveform_and_label(file_path):
      label = get_label(file_path)
      audio_binary = tf.io.read_file(file_path)
      waveform = decode_audio(audio_binary)
      return waveform, label
    waveform, label = get_waveform_and_label(sample_data)
    print(waveform.shape)
  # End Testing

  # yamnet start testing
  import os
  import soundfile as sf
  import params as yamnet_params
  import yamnet as yamnet_model
  from scipy.io import wavfile
  params = yamnet_params.Params()
  yamnet = yamnet_model.yamnet_frames_model(params)
  if not os.path.exists('yamnet.h5'):
    print('Error: curl -O https://storage.googleapis.com/audioset/yamnet.h5')
    exit()
  yamnet.load_weights('yamnet.h5')
  yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

  import pygame
  pygame.init()
  screen = pygame.display.set_mode((640, 480))
  font_header = pygame.font.Font(pygame.font.get_default_font(), 36)
  font = pygame.font.Font(pygame.font.get_default_font(), 36 * 2)

  text_surface = font.render('Hello world', True, (0, 0, 0))
  GRAY = (200, 200, 200)
  # yamnet end testing
  with recorder:
    last_detection = -1
    while not timed_out:
      audio_sample = recorder.get_audio(7921)[0]
      if False:
        wavfile.write('test.wav', 16000, audio_sample)
        wav_data, sr = sf.read('test.wav', dtype=np.int16)
      else:
        wav_data = np.array(audio_sample, dtype=np.int16)
        sr = AUDIO_SAMPLE_RATE_HZ
      assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
      waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
      waveform = waveform.astype('float32')
      # Convert to mono and the sample rate expected by YAMNet.
      if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
      if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)
      print('-------')
      # Predict YAMNet classes.
      scores, embeddings, spectrogram = yamnet(waveform)
      # Scores is a matrix of (time_frames, num_classes) classifier scores.
      # Average them along time to get an overall classifier output for the clip.
      prediction = np.mean(scores, axis=0)
      # Report the highest-scoring classes and their scores.
      top5_i = np.argsort(prediction)[::-1][:5]
      print(':\n' +
            '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
            for i in top5_i))
      print('{}:{:.3f}'.format(yamnet_classes[42], prediction[42]))
      print('{}:{:.3f}'.format(yamnet_classes[0], prediction[0]))
      print('{}:{:.3f}'.format(yamnet_classes[494], prediction[494]))

      target_predictions = prediction[42], prediction[0], prediction[494]
      target_classes = yamnet_classes[42], yamnet_classes[0], yamnet_classes[494]
      index = np.argsort(target_predictions)[::-1][0]
      black = (0, 0, 0)
      green = (0, 255, 0)
      red = (255, 0, 0)
      if index == 0:
        color = red
      elif index == 1:
        color = green
      else:
        color = black
      text1 = font.render(target_classes[index], True, color)
      header1 = font_header.render('R-zero Device Listening for Audio', True, (0, 0, 0))
      screen.fill(GRAY)
      screen.blit(header1, dest=(20, 100))
      screen.blit(text1, dest=(200, 200))
      pygame.display.update()
      '''
      line = '{}:{:.3f}'.format(yamnet_classes[42], prediction[42])
      label = Tk.Label(None, text=line, font=('Times', '18'), fg='blue')
      label.pack()
      label.mainloop()
      '''
      # End
      """
      spectrogram = feature_extractor.get_next_spectrogram(recorder).astype('float32')
      #spectrogram = feature_extractor.compute_spectrogram_and_normalize(waveform.numpy()[:15680], 16000)
      # plot_spectrogram(spectrogram)
      spectrogram = np.expand_dims(spectrogram, axis=-1)
      spectrogram = np.expand_dims(spectrogram, axis=0)
      input_details = interpreter.get_input_details()
      interpreter.set_tensor(input_details[0]['index'], spectrogram)
      # set_input(interpreter, spectrogram.flatten())
      interpreter.invoke()
      result = get_output(interpreter)
      # NOTE: Add softmax
      # NOTE: Remove negative label
      result = softmax(result)
      #print(result)
      if result_callback:
        result_callback(result, commands, labels)
      if dectection_callback:
        detection = -1
        if result[0] < negative_threshold:
          top3 = np.argsort(-result)[:3]
          for p in range(3):
            label = labels[top3[p]]
            if label not in commands.keys():
              continue
            if top3[p] and result[top3[p]] > commands[label]['conf']:
              if detection < 0:
                detection = top3[p]
        if detection < 0 and last_detection > 0:
          print("---------------")
          last_detection = 0
        if labels[detection] in commands.keys() and detection != last_detection:
          print(labels[detection], commands[labels[detection]])
          dectection_callback(commands[labels[detection]]['key'])
          last_detection = detection
      if spectrogram.mean() < 0.001:
        print("Warning: Input audio signal is nearly 0. Mic may be off ?")
      """
