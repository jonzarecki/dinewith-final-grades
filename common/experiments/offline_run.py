import datetime
import logging
import os
import sys

_using_debugger = getattr(sys, "gettrace", None)() is not None


def run_offline_if_requested():
    if "--offline" in sys.argv:
        if _using_debugger:
            logging.info("Won't run offline with debugger")
            print(f"current process PID = {os.getpid()}")
            return

        logging.info("performing offline run using nohup")

        curr_dir = os.path.dirname(sys.argv[0])
        outs_dir = f"{curr_dir}/offline_outs/"
        os.chdir(curr_dir)
        os.makedirs(outs_dir, exist_ok=True)
        outfilename = f"{outs_dir}/{datetime.datetime.now().isoformat(' ', 'seconds').replace(' ', '_')}.txt"
        arg_list = sys.argv.copy()
        arg_list.remove("--offline")
        os.system(
            "nohup sh -c '"
            + sys.executable
            + " -u "
            + " ".join(arg_list)
            + f"' 2> {outfilename} > {outfilename} & tail -f {outfilename}"
        )
        sys.exit(0)
    else:
        logging.info("no --offline, running normally")
        print(f"current process PID = {os.getpid()}")
        return
