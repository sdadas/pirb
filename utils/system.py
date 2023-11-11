import logging
import os
import subprocess
import platform
from pathlib import Path


def set_java_env():
    java_home = os.environ.get("JAVA_HOME", None)
    system = platform.system().lower()
    if java_home is None:
        logging.info("No JAVA_HOME found, trying to automatically guess path to Java")
        where_cmd = "where" if system == "windows" else "which"
        p = subprocess.Popen([where_cmd, "java"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result, err = p.communicate()
        if p.returncode != 0:
            raise IOError(err)
        res = result.decode("utf-8").strip()
        if system != "windows":
            res = os.path.realpath(res)
        res = str(Path(res).parent)
        os.environ["JAVA_HOME"] = res
