"""

Date: Jul 2018
Author: Aciel Eshky

"""

import os


def get_all_utterance_dirs(root_dir):
    list_of_utterance_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(fname.endswith('.ult') for fname in filenames):
            print(dirpath)
            list_of_utterance_dirs.append(dirpath)
    return list_of_utterance_dirs


def get_all_utterance_files(root_dir):
    list_of_utterance_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if filenames:
            for fname in filenames:
                if fname.endswith('.ult'):
                    list_of_utterance_files.append((dirpath, fname))
    return list_of_utterance_files


def get_dir_info(path):
    items = path.split("/")

    for i in items:
        if "uxtd" in i:
            dataset = "uxtd"
            speaker = items[-1]
            session = "Single"
            break
        elif "uxssd" in i:
            dataset = "uxssd"
            speaker = items[-2]
            session = items[-1]
        elif "upx" in i:
            dataset = "upx"
            speaker = items[-2]
            session = items[-1]
        elif "cleft" in i:
            dataset = "cleft"
            speaker = items[-2]
            session = items[-1]

    return {"dataset": dataset,
            "speaker": speaker,
            "session": session}


def get_extended_dir_info(path, filename):

    x = get_dir_info(path)

    return {"dataset": x["dataset"],
            "speaker": x["speaker"],
            "session": x["session"],
            "fbasename": filename.split(".")[0],
            "dirname": path}


def get_utterance_id(dataset, speaker, session, utterance):
    return dataset + "-" + speaker + "-" + session + "-" + utterance
