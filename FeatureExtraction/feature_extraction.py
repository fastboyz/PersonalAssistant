import argparse
import json
import math
import os

import librosa

features = {
    "mapping": [],
    "mfcc": [],
    "labels": []
}


def get_mfccs(dataset_path, json_path, n_mfcc, n_fft, hop_length, expected_num_mfccs_vectors, sample_rate):
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            semantic_label = dirpath.split(os.path.sep)[-1]
            features["mapping"].append(semantic_label)
            print("Processing: {}".format(semantic_label))

            for f in filenames:
                # Load the file

                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=sample_rate)

                mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc = mfcc.T

                if len(mfcc) == expected_num_mfccs_vectors:
                    print(f)
                    features["mfcc"].append(mfcc.tolist())
                    features["labels"].append(i-1)

    with open(json_path, 'w') as fp:
        json.dump(features, fp, indent=2)


def main(args):
    sample_per_track = args.duration * args.sample_rate
    expected_num_mfccs_vectors = math.ceil(sample_per_track / args.hop_length)
    get_mfccs(dataset_path=args.dataset_path, json_path=args.json_path, n_mfcc=args.n_mfcc, hop_length=args.hop_length,
              n_fft=args.n_fft, expected_num_mfccs_vectors=expected_num_mfccs_vectors, sample_rate=args.sample_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        This Script will split any audio file passed in parameter into chunks of a specified length
        """)

    parser.add_argument("--dataset_path", type=str, help="This is the path to the directory where the datasets reside",
                        required=True)
    parser.add_argument("--json_path", type=str, help="This is the path to the Resulting json folder", required=True)
    parser.add_argument("--duration", type=int, default=3, help="Length the audio files")
    parser.add_argument("--sample_rate", type=int, default=8000, help="The Audio files sample rate")
    parser.add_argument("--hop_length", type=int, default=512, help="The Hop length")
    parser.add_argument("--n_mfcc", type=int, default=13, help="Number of MFCCs")
    parser.add_argument("--n_fft", type=int, default=2048, help="Number of FTTs")

    args = parser.parse_args()
    main(args)
