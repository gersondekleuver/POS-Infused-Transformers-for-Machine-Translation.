from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import os
import torch


def read_txt(file):
    f = open(file, "r", encoding="utf-8")
    text = []
    data = f.read().splitlines()
    for line in data:
        text.append(line + "\n")
    return text


def save_txt(text, file):
    f = open(file, "w", encoding="utf-8")
    for line in text:
        f.write("".join(line)+"\n")
    f.close()
    # print("write file to:" + file)


def get_files(folders, lang):
    files = []
    for folder in folders:
        # read filenames from folder
        filenames = os.listdir(folder)
        for filename in filenames:
            if filename.endswith(f".{lang}"):
                files.append(folder+filename)
    return files


def create_source(folders, lang="pos.vt"):
    # Translate English to German
    # Write source file
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-vi-en")

    model.eval()

    files = get_files(folders, lang)
    for file in files:
        #  read file
        print("reading file: " + file)

        text = read_txt(file)

        # translate

        corrected = []
        all_unique_tokens = []

        # read 5 lines at a time
        count = 0
        for line in text:
            corrected_line = []
            words = line.split()
            for word in words:

                if word.startswith("Num"):
                    corrected_line.append("M")
                elif word.startswith("N"):
                    corrected_line.append("N")
                elif word.startswith("V"):
                    corrected_line.append("V")
                elif word.startswith("Adv"):
                    corrected_line.append("R")
                elif word.startswith("Adj"):
                    corrected_line.append("A")
                elif word.startswith("AUX"):
                    corrected_line.append("T")
                elif word.startswith("SC"):
                    corrected_line.append("C")
                elif word.startswith("CH"):
                    corrected_line.append("PCT")
                elif word.startswith("Pre"):
                    corrected_line.append("E")
                elif word.startswith("Pro"):
                    corrected_line.append("P")
                elif word.startswith("Prt"):
                    corrected_line.append("T")
                elif word.startswith("C"):
                    corrected_line.append("C")
                elif word.startswith("E"):
                    corrected_line.append("E")
                elif word.startswith("."):
                    corrected_line.append("PCT")

        # write file
        save_txt(corrected, file.replace(lang, "cor.pos.vt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",
                        default="./data-bin/VN-tree-banks/train/")
    parser.add_argument("--valid",
                        default="./data-bin/VN-tree-banks/valid/")
    parser.add_argument("--test",
                        default="./data-bin/VN-tree-banks/test/")

    args = parser.parse_args()

    folders = [args.train, args.valid, args.test]

    create_source(folders)
