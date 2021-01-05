""" This Script is used for the data collection part of our problem.
    Use cases:

    1- Record background noise. --> Record until stopped by the user or for a certain amount of time
    2- Record the key word --> let the user start recording a certain amount of times
"""

import argparse
import os
import time
import wave

import pyaudio


class DataCollector:
    """
    Constructor, prepares the stream for recording.
    """

    def __init__(self, args):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = args.rate
        self.CHUNK = 1024
        self.RECORD_SECONDS = args.record_seconds

        self.audio = pyaudio.PyAudio()

        self.audio_stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True,
                                            frames_per_buffer=self.CHUNK, output=True)

    """
    Save Audio to file
    """

    def save_to_file(self, frames, file_name):
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio.terminate()

        file = wave.open(file_name, 'wb')  # wb -> write byte
        file.setnchannels(self.CHANNELS)
        file.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        file.setframerate(self.RATE)
        file.writeframes(b''.join(frames))
        file.close()


def record_key_word_mode(args):
    index = 0
    try:
        input("Press Enter to start.\r\n"
              "Each Recordings will be for {} seconds \r\n"
              "Press ctl+c to stop recording".format(args.record_seconds))
        while True:
            data_collector = DataCollector(args)
            frames = []
            print("Recording...")
            for i in range(int(data_collector.RATE / data_collector.CHUNK) * data_collector.RECORD_SECONDS):
                data = data_collector.audio_stream.read(data_collector.CHUNK, exception_on_overflow=False)
                frames.append(data)

            print("Saving...")
            file_path = os.path.join(args.keyword_file_path, "{}.wav".format(index))
            data_collector.save_to_file(frames=frames, file_name=file_path)
            index += 1
    except KeyboardInterrupt:
        print("User Interruption")
    except Exception as ex:
        print(str(ex))


def default_mode(args):
    data_collector = DataCollector(args)
    frames = []
    print("Recording until Stopped...\r\n"
          "Press ctl+c to stop")
    try:
        while True:
            data = data_collector.audio_stream.read(data_collector.CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        print("User Interruption")
    except Exception as ex:
        print(str(ex))
    data_collector.save_to_file(frames=frames, file_name=args.file_path)


def limited_time_mode(args):
    data_collector = DataCollector(args)
    frames = []
    try:
        while True:
            print("Recording for {} seconds".format(args.record_seconds))
            time.sleep(0.1)
            for i in range(int(data_collector.RATE / data_collector.CHUNK) * data_collector.RECORD_SECONDS):
                data = data_collector.audio_stream.read(data_collector.CHUNK, exception_on_overflow=False)
                frames.append(data)
            raise Exception("Done")
    except KeyboardInterrupt:
        print("User Interruption")
    except Exception as ex:
        print(str(ex))
    data_collector.save_to_file(frames=frames, file_name=args.file_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
    This script is used to Record audio files.
    
    There is 3 different modes
      
    Mode 1: Records until the receives the ctl+c Command to stop 
    
    Mode 2: Records for a certain amount of time
    
    Mode 3: Records for a certain amount of time for a certain number of time (Keyword Mode)
    """)

    parser.add_argument("--rate", type=int, default=8000, help="This is the Sample rate of the recording")
    parser.add_argument("--record_seconds", type=int, default=None, help="this is the Time/length of the recording")
    parser.add_argument("--file_path", type=str, default=None,
                        help="this is path where to save the recordings /path/to/file.wav")
    parser.add_argument("--keyword_file_path", type=str, default=None,
                        help="this is path where to save the recordings /path/to")
    parser.add_argument("--keyword_mode", default=False, help="Set to Keyword Recording mode", action="store_true")

    args = parser.parse_args()

    if args.record_seconds is None:
        if args.file_path is None:
            raise Exception("Please set the --file_path before continuing")
        default_mode(args)
    elif args.record_seconds is not None and args.keyword_mode is False:
        if args.file_path is None:
            raise Exception("Please set the --file_path before continuing")
        limited_time_mode(args)
    elif args.record_seconds is not None and args.keyword_mode is True:
        if args.keyword_file_path is None:
            raise Exception("Please set the --keyword_file_path before continuing")
        record_key_word_mode(args)
