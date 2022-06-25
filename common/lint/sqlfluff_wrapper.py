import re
import subprocess
from typing import List

from common.constants import PROJECT_ROOT


def _insert_line_numbers(txt: str) -> str:
    return "\n".join([f"{n + 1:03d} {line}" for n, line in enumerate(txt.split("\n"))])


def _normalize_line_endings(txt: str) -> str:
    return txt.replace("\r\n", "\n").replace("\r", "\n")


def _remove_tailing_newline(txt: str) -> str:
    return re.sub(r"\n\s*$", "", txt)


def _call_shell_get_stdout(command_args: List[str], input_str: str) -> str:
    return _remove_tailing_newline(
        _normalize_line_endings(
            subprocess.run(
                command_args,
                check=False,
                cwd=PROJECT_ROOT,  # change cwd to root in order for sqlfluff to detect setup.cfg
                input=str.encode(input_str),
                capture_output=True,
                shell=True,
            ).stdout.decode()
        )
    ).replace(
        "\x1b[0m", ""
    )  # remove 'color-reset' char


def apply_sqlfluff(query: str, dialect: str = "ansi", do_print: bool = False) -> str:
    """Apply sqlfluff `lint` then `fix` on a query. Returns output.

    SQLFluff is configured in setup.cfg in the main project path.

    Args:
        query: SQL query to be checked
        dialect: optional dialect of the sql query
        do_print: boolean stating we want to print the output

    Returns:
        Commands output (with some formatting)
    """
    linter_output = _call_shell_get_stdout(["sqlfluff", "lint", "--dialect", dialect, "-"], query)

    linter_output = re.sub(r"^==.*\n", "", linter_output)
    linter_output = re.sub(r"\n*.*All Finished.*\n*$", "", linter_output)

    if linter_output != "":
        fixed_query = _call_shell_get_stdout(["sqlfluff", "fix", "--dialect", dialect, "-"], query)

        linter_output = (
            f"SqlFluff errors for: \n{_insert_line_numbers(query)}\n"
            f"{linter_output}\n"
            f"Fixed sql: \n{_insert_line_numbers(fixed_query)} "
        )

    if do_print:
        print(linter_output)

    return linter_output
