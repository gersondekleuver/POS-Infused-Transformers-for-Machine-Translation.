import time
import nltk
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import os
import argparse


def train_tokenizer(folders, lang):
    # load Pretrained tokenizer
    if os.path.exists(f"./data-bin/IWSLT_EN_VT/vocab_{lang}.json"):
        tokenizer = Tokenizer.from_file(
            f"./data-bin/IWSLT_EN_VT/vocab_{lang}.json")

    else:
        # Train tokenizer
        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False)

        trainer = trainers.BpeTrainer(
            vocab_size=40000, special_tokens=["<|endoftext|>"])
        files = get_files(folders, lang)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.train(
            files, trainer=trainer)
        tokenizer.decoder = decoders.ByteLevel()
        # Save tokenizer
        tokenizer.save(f"./data-bin/IWSLT_EN_VT/vocab_{lang}.json")

    return tokenizer


def POS_tag(text, lang):
    # POS tag sentences
    pos_tagged = []
    for line in text:
        pos_tagged.append(nltk.pos_tag(line, lang=lang))
    return pos_tagged


def get_files(folders, lang):
    files = []
    for folder in folders:
        # read filenames from folder
        filenames = os.listdir(folder)
        for filename in filenames:
            if filename.endswith(f".{lang}"):
                files.append(folder+filename)
    return files


def get_bpe_files(folder, lang):
    files = []
    # read filenames from folder
    filenames = os.listdir(folder)
    for filename in filenames:
        if filename.endswith(f".{lang}"):
            files.append(folder+filename)
    return files


def get_bpe_files2(folder, lang):
    files = []
    # read filenames from folder
    filenames = os.listdir(folder)
    for filename in filenames:
        if filename.endswith(f".{lang}"):
            files.append(folder+filename)
    return files


def read_txt(file):
    f = open(file, "r", encoding="utf-8")
    text = []
    data = f.read().splitlines()
    for line in data:
        text.append(line)
    return text


def save_txt(text, file):
    f = open(file, "w", encoding="utf-8")
    for line in text:
        f.write(" ".join(line)+"\n")
    f.close()
    # print("write file to:" + file)


def get_raw(text):
    # recover sentences from bpe sentences
    clean = []

    for line in text:

        clean_line = []
        Line_index = 0

        line = line.split(" ")
        while Line_index < len(line):

            if "@@" in line[Line_index]:

                Upper_limit = Line_index + 1
                string = ""
                switch = True

                while "@@" in line[Upper_limit] and switch == True:
                    if Upper_limit >= len(line)-1:
                        switch = False
                    else:
                        Upper_limit += 1

                Upper_limit += 1

                for i in range(Line_index, Upper_limit):
                    string += line[i]

                clean_line.append(string.replace(
                    " ", "").replace("@@", ""))
                Line_index = Upper_limit

            else:
                clean_line.append(line[Line_index])
                Line_index += 1

        clean.append(clean_line)

    return clean


def pos_subid(pos_tag, ids, limit_dict):
    a = int(limit_dict[pos_tag])
    if a == 0:
        return pos_tag
    elif ids < a:
        return pos_tag+str(ids)
    else:
        return pos_tag+str(a)


def tokenize(folders, tokenizer, lang):
    # tokenize sentences
    files = get_files(folders, lang)

    for file in files:

        tokenized = []
        text = read_txt(file)

        for i, line in enumerate(text):
            # print(f"{i/len(text)*100:.2f}%", end="\r")
            tokenized.append(tokenizer.encode(line).tokens)
        save_txt(
            tokenized, f"./data-bin/IWSLT_EN_VT/{file.split('/')[-1].split('.')[0]}.bpe.{file.split('/')[-1].split('.')[1]}")

    return tokenized


def get_limit_dict(limit_data):
    l_keys = []
    l_ids = []
    for line in limit_data:
        line = line.split(" ")
        l_keys.append(line[0])
        l_ids.append(line[1])
    return dict(zip(l_keys, l_ids))


def get_UTS_pos(bpe_data, raw_text, limit_dict):
    """
    bpe_data = list of bpe sentences
    raw_text = list of raw sentences
    """
    pos_test = []
    time1 = time.time()
    for i in range(len(bpe_data)):
        if i % 100000 == 1:
            print(i)
            print(time.time()-time1)
            print("#"*66)
            time1 = time.time()
        pos = []
        line = bpe_data[i].split(" ")
        pos_line = []

        for word in raw_text[i]:
            try:
                pos_line.append(UTS_pos_tag(word)[0])

            except:
                pos_line.append(("", "UNK"))
        p_line = []

        for w, p in pos_line:

            if w == p or w in [":", "?", "-", "...", ";", "--", "!"] or p in ["(", ")", "``"]:
                p_line.append("PCT")
            else:
                # combining "(",")"and"``" are named as PCT,
                # "''"and"$" are named as SYM$, WP$ is combined to WP
                if p in ["$", "''", "SYM"]:
                    p_line.append("SYM$")
                elif p == "WP$":
                    p_line.append("WP")
                else:
                    p_line.append(p)

        extended_pos_tags = []
        Pos_index = 0
        Line_index = 0

        while Line_index < len(line):

            if "Ġ" in line[Line_index]:
                if Line_index == len(line)-1:

                    extended_pos_tags.append(
                        p_line[Pos_index])
                    Line_index += 1
                    Pos_index += 1

                elif "Ġ" in line[Line_index+1]:

                    extended_pos_tags.append(
                        p_line[Pos_index])
                    Line_index += 1
                    Pos_index += 1

                else:

                    Upper_limit = Line_index + 1

                    string = ""
                    switch = True

                    while "Ġ" not in line[Upper_limit] and switch == True:
                        if Upper_limit >= len(line)-1:
                            switch = False
                        else:
                            Upper_limit += 1

                    POS_count = 0
                    for _ in range(Line_index, Upper_limit):

                        extended_pos_tags.append(
                            f"{p_line[Pos_index]}" + f"{POS_count}")
                        Line_index += 1
                        POS_count += 1

                    Pos_index += 1

            else:
                extended_pos_tags.append(
                    p_line[Pos_index])
                Line_index += 1
                Pos_index += 1

        assert len(extended_pos_tags) == len(line)
        pos_test.append(extended_pos_tags)
    return pos_test


def get_nltk_pos(bpe_data, raw_text):
    """
    bpe_data = list of bpe sentences
    raw_text = list of raw sentences
    """

    pos_test = []
    time1 = time.time()
    for i in range(len(bpe_data)):
        if i % 100000 == 1:
            print(i)
            print(time.time()-time1)
            print("#"*66)
            time1 = time.time()
        pos = []
        line = bpe_data[i].split(" ")

        pos_line = nltk.pos_tag(raw_text[i])
        p_line = []
        for w, p in pos_line:
            if w == p or w in [":", "?", "-", "...", ";", "--", "!"] or p in ["(", ")", "``"]:
                p_line.append("PCT")
            else:
                # combining "(",")"and"``" are named as PCT,
                # "''"and"$" are named as SYM$, WP$ is combined to WP
                if p in ["$", "''", "SYM"]:
                    p_line.append("SYM$")
                elif p == "WP$":
                    p_line.append("WP")
                else:
                    p_line.append(p)

        extended_pos_tags = []
        Pos_index = 0
        Line_index = 0

        while Line_index < len(line):

            if "@@" in line[Line_index]:

                Upper_limit = Line_index + 1

                switch = True

                while "@@" in line[Upper_limit] and switch == True:
                    if Upper_limit >= len(line)-1:
                        switch = False
                    else:
                        Upper_limit += 1

                Upper_limit += 1

                POS_count = 0
                for _ in range(Line_index, Upper_limit):

                    extended_pos_tags.append(
                        f"{p_line[Pos_index]}" + f"{POS_count}")
                    Line_index += 1
                    POS_count += 1

                Pos_index += 1

            else:
                extended_pos_tags.append(
                    p_line[Pos_index])
                Line_index += 1
                Pos_index += 1

        assert len(extended_pos_tags) == len(line)
        pos_test.append(extended_pos_tags)
    return pos_test


def get_universal_pos(bpe_data, raw_pos):
    """
    bpe_data = list of bpe sentences
    raw_pos = list of pos tags
    """
    pos_test = []
    errors = 0

    time1 = time.time()
    for i in range(len(bpe_data)):
        if i % 100000 == 1:
            print(i)
            print(time.time()-time1)
            print("#"*66)
            time1 = time.time()
        pos = []
        bpe_line = bpe_data[i].split(" ")

        p_line = raw_pos[i].split(" ")

        extended_pos_tags = []
        pos_index = 0
        Line_index = 0

        while Line_index < len(bpe_line):

            if "Ġ" in bpe_line[Line_index] or Line_index == 0:
                if Line_index == len(bpe_line)-1:

                    extended_pos_tags.append(
                        p_line[pos_index])

                    try:
                        extended_pos_tags.append(
                            p_line[pos_index])

                    except:
                        errors += 1
                        print(f"error at line {i}, total: {errors}")
                        Line_index = len(bpe_line)
                        continue

                    Line_index += 1
                    pos_index += 1

                elif "Ġ" in bpe_line[Line_index+1]:

                    try:
                        extended_pos_tags.append(
                            p_line[pos_index])

                    except:
                        errors += 1
                        print(f"error at line {i}, total: {errors}")
                        Line_index = len(bpe_line)
                        continue

                    Line_index += 1
                    pos_index += 1

                else:

                    Upper_limit = Line_index + 1

                    string = ""
                    switch = True

                    while "Ġ" not in bpe_line[Upper_limit] and switch == True:
                        if Upper_limit >= len(bpe_line)-1:
                            switch = False
                        else:
                            Upper_limit += 1

                    POS_count = 0
                    for _ in range(Line_index, Upper_limit):

                        extended_pos_tags.append(
                            f"{p_line[pos_index]}" + f"{POS_count}")
                        Line_index += 1
                        POS_count += 1

                    pos_index += 1

            else:
                extended_pos_tags.append(
                    p_line[pos_index])
                Line_index += 1
                pos_index += 1

        pos_test.append(extended_pos_tags)
    return pos_test


parser = argparse.ArgumentParser()

parser.add_argument("--bpe", default="./data-bin/IWSLT_EN_VT/")
parser.add_argument("--limit", default="./data-bin/pos_limit100.txt")
parser.add_argument("--lang", default="en")
parser.add_argument("-t", "--tokenize",  default=True,
                    action='store_true')
parser.add_argument("-n", "--NLTK", default=True, action='store_true')
parser.add_argument("-s", "--UTS", default=False, action='store_true')
parser.add_argument("-u", "--universal", default=False, action='store_true')

args = parser.parse_args()

lang = args.lang
bpe_folder = args.bpe
limit_data = read_txt(args.limit)


if args.tokenize:
    tokenizer = train_tokenizer(bpe_folder, lang)
    print(f"Tokenizing {bpe_folder} in {lang}")
    tokenize(bpe_folder, tokenizer, lang)


if not args.universal:

    if args.NLTK:
        files = get_bpe_files(bpe_folder, lang)
        print(files)
        for file in files:
            print(f"load bpe {file}")
            bpe_file = read_txt(file)
            print("get pos tags")
            raw_file = get_raw(bpe_file)
            pos_file = get_nltk_pos(bpe_file, raw_file)
            save_txt(pos_file, file+".pos")

    if args.UTS:

        limit_dict = get_limit_dict(limit_data)
        files = get_bpe_files2(bpe_folder, lang)
        print(files)
        for file in files:
            print(f"load bpe {file}")
            bpe_file = read_txt(file)
            print("preprocess bpe sentences")
            raw_file = process_bpe(bpe_file, tokenizer)
            print("get pos tags")
            pos_file = get_UTS_pos(bpe_file, raw_file, limit_dict)
            save_txt(pos_file, file+".pos")


elif args.NLTK and args.universal:

    bpe_folder = "/".join(folder.split("/")
                          [:-2]) + f"/bpe/"
    bpe_files = get_bpe_files2(bpe_folder, lang)

    for bpe_file in bpe_files:

        raw_pos_file = "/".join(bpe_file.split("/")
                                [:-2]) + f"/convert/" + bpe_file.split("/")[-1].split(".")[0] + ".en.pos"

        print(f"load bpe file {bpe_file}")
        bpe = read_txt(bpe_file)
        print(f"load raw pos file {raw_pos_file}")
        raw_pos = read_txt(raw_pos_file)
        print("format the universal pos tags")
        pos_file = get_universal_pos(bpe, raw_pos)

        print(f"saving pos tags at {bpe_file+'.pos'}")
        save_txt(pos_file, bpe_file+".pos")
