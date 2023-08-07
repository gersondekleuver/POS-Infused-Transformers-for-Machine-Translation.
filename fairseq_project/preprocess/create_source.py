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


def create_source(folders, lang="vn"):
    # Translate English to German
    # Write source file
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-vi-en")

    model.eval()

    files = get_files(folders, lang)
    for file in files:
        #  read file

        text = read_txt(file)

        # translate

        translated = []

        # read 5 lines at a time
        count = 0
        for i in range(0, len(text), 6):

            batch = text[i:i+6]

            batch = tokenizer.prepare_seq2seq_batch(batch, return_tensors="pt")
            translated_batch = model.generate(**batch)
            translated_batch = tokenizer.batch_decode(
                translated_batch, skip_special_tokens=True)

            translated.extend(translated_batch)

        # save translated file
        save_txt(translated, file.replace(lang, "en"))


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
