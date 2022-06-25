import copy
import sys
from typing import List, Tuple

import yaml


def extract_names_from_dict(d: dict, prefix="") -> List[Tuple[str, str]]:
    all_tpls = []
    for key, val in d.items():
        if prefix != "":
            name = f"{prefix}-{key}"
        else:
            name = key
        if isinstance(val, dict):
            all_tpls += extract_names_from_dict(val, name)
        elif isinstance(val, list):
            all_tpls.append((name, [str(v) for v in val]))
        else:
            all_tpls.append((name, str(val)))

    return all_tpls


def convert_config_to_args():
    if "--config" in sys.argv:
        cfg_idx = sys.argv.index("--config")
        yaml_cfg_path = sys.argv[cfg_idx + 1]
        with open(yaml_cfg_path) as f_in:
            cfg = yaml.safe_load(f_in)

        all_configs = extract_names_from_dict(cfg)

        # remove --config from argv
        new_argv = []
        for i, itm in enumerate(sys.argv):
            if i not in (cfg_idx, cfg_idx + 1):
                new_argv.append(itm)
        sys.argv = new_argv

        # add configs from file
        for c_name, c_val in all_configs:
            if f"--{c_name}" not in sys.argv:
                sys.argv.append(f"--{c_name}")
                if isinstance(c_val, list):
                    sys.argv += c_val
                else:
                    sys.argv.append(c_val)


def make_config_loggable(args):
    clean_args = copy.copy(args)
    for k, v in args.__dict__.items():
        if isinstance(v, (int, str, bool)):
            continue
        if isinstance(v, list):
            if all(isinstance(itm, int) for itm in v):
                clean_args.__dict__[k] = str(v)
                continue
        del clean_args.__dict__[k]

    return clean_args
