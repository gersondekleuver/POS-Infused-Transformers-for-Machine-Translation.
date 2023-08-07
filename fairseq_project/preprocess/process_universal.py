import os
import sys
import argparse


def convert_from_conllu(input_files, lang):

    for input_file in input_files:

        pr_file = "/".join(input_file.split("/")
                           [:-2]) + f"/convert/" + input_file.split("/")[-1]

        pos_file = "." + pr_file.split(".")[1] + f".pos.{lang}"
        text_file = "." + pr_file.split(".")[1] + f".{lang}"

        with open(input_file, 'r', encoding='utf-8') as f:
            newline = 1
            prev_hyph = 0
            lines = f.readlines()

            with open(pos_file, "w", encoding='utf-8') as pos:
                with open(text_file, 'w', encoding='utf-8') as text:
                    for line in lines:
                        if line.startswith('#'):
                            continue

                        if line.startswith("\n"):
                            pos.write("\n")
                            text.write("\n")

                            newline = 1
                            continue

                        line = line.split("\t")

                        if "-" in line[0]:
                            continue

                        words = line[1].split(" ")

                        for word in words:
                            postag = line[4]

                            if word == "." or word == "!" or word == "?":

                                text.write(word)
                                pos.write("PCT")

                                newline = 1

                            elif word == "," or word == ":" or word == ";" or word == "”":

                                text.write(word)
                                pos.write("PCT")

                            elif word == "“":

                                text.write(word)
                                pos.write("PCT")

                            elif word == "(" or word == "[" or word == "{":

                                text.write(word)
                                pos.write("LRB")

                            elif word == ")" or word == "]" or word == "} ":

                                text.write(word)
                                pos.write("RRB")

                            elif word == "’" or word == "'":

                                text.write(word)
                                pos.write("POS")

                            elif word == "‘":

                                text.write(word)
                                pos.write("POS")

                            elif word == "-":
                                text.write(word)
                                pos.write("HYPH")

                            elif word == "\"":

                                text.write(word)
                                pos.write("PCT")

                            else:

                                text.write(word)
                                pos.write(postag)

                            if newline:
                                newline = 0

                            text.write(" ")
                            pos.write(" ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    input_files = ["./data-bin/VN-tree-banks/train/vi_vtb-ud-train.conllu",
                   "./data-bin/VN-tree-banks/valid/vi_vtb-ud-dev.conllu",
                   "./data-bin/VN-tree-banks/test/vi_vtb-ud-test.conllu"]

    convert_from_conllu(input_files, "vt")
