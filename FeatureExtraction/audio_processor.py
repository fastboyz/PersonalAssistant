"""
Process audio files
"""
import argparse
import math
import os

from pydub import AudioSegment
from pydub.utils import make_chunks

"""
Main Issue: all recording don't have the same length 
Sam File ->5 sec  but in general 2-3 sec 
Noise file -> 50 min+ 
common voice file -> 2-8 sec


Pydub -> make_chunk that make chunkc of audio files
"""


def split_audio_file(file, seconds, save_path):
    audio_segement = AudioSegment.from_file(file)

    length = seconds * 1000
    chunks = make_chunks(audio_segement, length)

    second_of_silence = AudioSegment.silent(duration=math.fabs(math.ceil(seconds) * 1000) - length)

    print("name: {} size: {}".format(file.split(os.path.sep)[-1], len(audio_segement)))
    for i, chunk in enumerate(chunks):
        if len(chunk) == length:
            chunk = chunk + second_of_silence
            og_file_name = file.split(os.path.sep)[-1]
            new_file_name = "{}_{}".format(i, og_file_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = os.path.join(save_path, new_file_name)
            chunk.export(file_path, format="wav")


def dir_mode(dir, seconds, save_path):
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dir)):
        for file in filenames:
            split_audio_file(os.path.join(dirpath, file), seconds=seconds, save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        This Script will split any audio file passed in parameter into chunks of a specified length
        """)

    parser.add_argument("--file", type=str, default=None, help="This is the path to the file you are trying to split", )
    parser.add_argument("--seconds", type=float, default=2, help="Length of each chunks")
    parser.add_argument("--save_path", type=str,
                        help="This is path where to save the chunks", required=True)
    parser.add_argument("--dir", type=str, default=None, help="This is the path to the file you are trying to split", )

    args = parser.parse_args()

    if args.file is not None:
        split_audio_file(args.file, args.seconds, args.save_path)
    elif args.dir is not None:
        dir_mode(args.dir, args.seconds, args.save_path)
    elif args.file is None and args.dir is None:
        raise Exception("Please set either the --file or --dir to run the Script")
