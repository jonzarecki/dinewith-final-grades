import errno
import os
import pathlib
import random
import shutil
import sys
import tempfile
from distutils.dir_util import copy_tree
from typing import List

from common.constants import PROJECT_ROOT, STATE_DIR, _curr_time

from experiments.tee import Tee

using_debugger = getattr(sys, "gettrace", None)() is not None


def save_all_py_files(folder_path, save_output=True):
    if using_debugger:
        return _curr_time
    runfile_name = os.path.basename(sys.argv[0])
    python_files_path = os.path.join(STATE_DIR, folder_path, runfile_name, _curr_time, "project_files")
    os.makedirs(python_files_path, exist_ok=True)

    if save_output:
        t = Tee(f"{python_files_path}/output.txt")
        t.__enter__()

    for ext in ["py", "yaml"]:  # copy all files with relevant extension
        python_files_in_dir = list_all_files_in_folder(PROJECT_ROOT, ext, recursively=True)
        copy_files_while_keeping_structure(python_files_in_dir, PROJECT_ROOT, python_files_path)
    with open(f"{python_files_path}/runscript.txt", "w") as f:
        f.write(" ".join([runfile_name] + sys.argv[1:]))
    print(f"files saved in: \n {python_files_path}")
    return _curr_time


def shuffle_rows(in_filename, out_filename):
    """
    shuffles the rows of $in_filename and puts the output in $out_filename
    :param in_filename: file name of the input file
    :type in_filename: str
    :param out_filename: file name of the output file
    :type out_filename: str
    :return: None
    """
    with open(in_filename) as source:
        data = [(random.random(), line) for line in source]
    data.sort()
    with open(out_filename, "w") as target:
        for _, line in data:
            target.write(line)


def merge_similar_rows(in_filename, out_filename):
    """
    merges exact rows in $in_filename and puts the output file in $out_filename
    :param in_filename: file name of the input file
    :type in_filename: str
    :param out_filename: file name of the output file
    :type out_filename: str
    :return: None
    """
    lines_seen = set()  # holds lines already seen
    out_lines = []
    with open(in_filename) as in_file:
        for line in in_file:
            if line not in lines_seen:  # not a duplicate
                out_lines.append(line)
                lines_seen.add(line)
    with open(out_filename, "w") as out_file:
        for line in out_lines:
            out_file.write(line)


def create_temp_folder(prefix=None):
    """
    creates a new temporary directory and returns it's path
    :param prefix: the prefix for the temp folder
    :return: full path of the new directory
    """
    if prefix is not None:
        return tempfile.mkdtemp(prefix=prefix)
    else:
        return tempfile.mkdtemp()


def copy_folder_contents(src_dir, dst_dir):
    # type: (str, str) -> None
    """
    copies all files from one directory to another
    :param src_dir: path to src directory
    :param dst_dir: path to dst directory
    :return: None
    """
    assert src_dir != dst_dir, "src and dst directories shouldn't be the same, check code"
    copy_tree(src_dir, dst_dir)


def delete_folder_with_content(folder_path):
    # type: (str) -> None
    """
    Deletes a folder recursively with all it's contents (no warnings)
    DANGEROUS USE WITH CARE
    :param folder_path: The absolute path to folder
    :return: None
    """
    shutil.rmtree(folder_path)


def makedirs(folder_path, exists_ok=True):
    # type: (str, bool) -> None
    """
    Create all folders in the path, doesn't fail of exists_ok is True
    :param folder_path: the absolute path to the folder
    :param exists_ok: states if we should fail when the folder already exists
    :return: $folder_path
    """
    if exists_ok:
        try:
            os.makedirs(folder_path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
    else:
        os.makedirs(folder_path)


def list_all_files_in_folder(fold_abspath, file_ext, recursively=True) -> List[str]:
    if recursively:
        file_list = list(pathlib.Path(fold_abspath).glob("**/*." + file_ext))
    else:
        file_list = list(pathlib.Path(fold_abspath).glob("*." + file_ext))
    return [str(p) for p in file_list]


def copy_file(file_path, dst_dir):
    shutil.copy(file_path, dst_dir)


def copy_files_while_keeping_structure(files_path_list, orig_dir, dst_dir):
    # copy all files to their correct structure in dst_dir
    for file_path in files_path_list:
        file_dst_parent = os.path.join(dst_dir, str(pathlib.PosixPath(file_path).relative_to(orig_dir).parent))
        makedirs(file_dst_parent)
        copy_file(file_path, file_dst_parent)


def readlines(file_path):
    with open(file_path) as f:
        file_lines = f.read().splitlines()
    return file_lines
