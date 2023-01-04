import os

if __name__ == '__main__':
    directorys = ["cv-other-dev/cv-other-dev", "cv-other-test/cv-other-test",
                  "cv-other-train/cv-other-train", "cv-valid-dev/cv-valid-dev",
                  "cv-valid-test/cv-valid-test", "cv-valid-train/cv-valid-train"]
    for directory in directorys:
        for filename in os.listdir(directory):
            if '.mp3' in filename:
                f = os.path.join(directory, filename)
                newName = f.replace('.mp3', '.wav')
                print(f'name =  {f}')
                print(f'newName = {newName}')
                os.system(f"ffmpeg -i {f} {newName}")
                os.system(f"del {f}")

    for directory in directorys:
        for filename in os.listdir(directory):
            if '.mp3' in filename:
                os.remove(directory + "/" + filename)