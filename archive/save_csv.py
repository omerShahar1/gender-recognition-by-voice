import glob
import pandas as pd


if __name__ == '__main__':
    files = glob.glob("../data/*.csv")
    df2 = pd.read_csv("../data/cv-other-dev.csv")
    df3 = pd.read_csv("../data/cv-other-test.csv")
    df4 = pd.read_csv("../data/cv-other-train.csv")
    df5 = pd.read_csv("../data/cv-valid-dev.csv")
    df6 = pd.read_csv("../data/cv-valid-test.csv")
    df7 = pd.read_csv("../data/cv-valid-train.csv")

    df = pd.concat([df2, df3, df4, df5, df6, df7], ignore_index=True)
    df.to_csv("temp_file.csv")

    temp = open("temp_file.csv", "r")
    temp = ''.join([i for i in temp]).replace("mp3", "npy").replace("wav", "npy")

    final_file = open("../balanced_all.csv", "w")
    final_file.writelines(temp)
    final_file.close()
