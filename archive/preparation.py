import glob
import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm



def extract_feature(file_name):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    X, sample_rate = librosa.load(file_name)

    stft = np.abs(librosa.stft(X))
    result = np.array([])

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma))

    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, contrast))

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    result = np.hstack((result, tonnetz))
    return result


if __name__ == '__main__':
    dirname = "data"

    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    csv_files = glob.glob("*.csv")

    for j, csv_file in enumerate(csv_files):
        print("[+] Preprocessing", csv_file)
        df = pd.read_csv(csv_file)
        # take filename, gender and age columns
        new_df = df[["filename", "age", "gender"]]

        print("Previously:", len(new_df), "rows")
        # take only people with written age (i.e dropping invalid and others)
        # new_df = new_df[np.logical_or(new_df['age'] != "", new_df['age'] != "")]

        new_df = new_df[
            np.logical_and(np.logical_or(new_df["gender"] == "male", new_df["gender"] == "female"),
            np.logical_or(new_df["age"] == "teens",
                np.logical_or(new_df["age"] == "twenties",
                    np.logical_or(new_df["age"] == "thirties",
                        np.logical_or(new_df["age"] == "fourties",
                            np.logical_or(new_df["age"] == "fifties",
                                np.logical_or(new_df["age"] == "sixties",
                                    np.logical_or(new_df["age"] == "seventies",
                                        np.logical_or(new_df["age"] == "eighties", new_df["age"] == "nineties")))))))))]
        print("Now:", len(new_df), "rows")

        new_csv_file = os.path.join(dirname, csv_file)
        # save new preprocessed CSV
        new_df.to_csv(new_csv_file, index=False)
        # get the folder name
        folder_name, _ = csv_file.split(".")
        audio_files = glob.glob(f"{folder_name}/{folder_name}/*")
        all_audio_filenames = set(new_df["filename"])
        for i, audio_file in tqdm(list(enumerate(audio_files)), f"Extracting features of {folder_name}"):
            splited = os.path.split(audio_file)
            # audio_filename = os.path.join(os.path.split(splited[0])[-1], splited[-1])
            audio_filename = f"{os.path.split(splited[0])[-1]}/{splited[-1]}"
            # print("audio_filename:", audio_filename)
            if audio_filename in all_audio_filenames:
                # print("Copyying", audio_filename, "...")
                src_path = f"{folder_name}/{audio_filename}"
                target_path = f"{dirname}/{audio_filename}"
                # create that folder if it doesn't exist
                if not os.path.isdir(os.path.dirname(target_path)):
                    os.mkdir(os.path.dirname(target_path))
                features = extract_feature(src_path)
                target_filename = target_path.split(".")[0]
                np.save(target_filename, features)
