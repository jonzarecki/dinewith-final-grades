# pylint: disable=import-outside-toplevel,reimported,redefined-outer-name,invalid-name
import itertools
import logging
import math
import multiprocessing
import sys
import traceback
from typing import Any, Callable, Iterable, Iterator, List, Sized, Tuple

from pathos.pools import ProcessPool as Pool
from tqdm.autonotebook import tqdm

from common.itertools_ext import flatten

i = 0
proc_count = 1
force_serial = False


class LoggerWriter:
    def __init__(self, level: Callable = logging.debug):  # type: ignore
        self.level = level

    def write(self, message: str) -> None:
        if message != "\n":
            self.level(message)

    def flush(self) -> None:
        self.level(sys.stderr)


def _chunk_spawn_fun(args_list: Iterable[Tuple[Any, Callable[[Any], Any], int, bool]]) -> List[Any]:
    return [_spawn_fun(args) for args in args_list]


def _spawn_fun(args: Tuple[Any, Callable[[Any], Any], int, bool]) -> Any:
    """Internal function that spawned tqdm worked start from."""
    f_input, func, proc_i, keep_child_tqdm = args
    import random  # pylint: disable=import-outside-toplevel
    import sys  # pylint: disable=import-outside-toplevel

    import numpy  # pylint: disable=import-outside-toplevel

    old_err = sys.stderr
    if not keep_child_tqdm:
        sys.stderr = LoggerWriter()  # type: ignore
    random.seed(1554 + proc_i)
    numpy.random.seed(42 + proc_i)  # set random seeds
    global force_serial
    force_serial = True

    try:
        res = func(f_input)
        return res
    except:  # noqa
        traceback.print_exc(file=sys.stdout)
        raise  # re-raise exception
    finally:
        sys.stderr = old_err


def chunk_iterator(itr: Iterable[Any], chunk_size: int) -> Iterator[List[Any]]:
    """Returns the values of the iterator in chunks of size $chunk_size.

    Args:
        itr: The iterator we want to split into chunks
        chunk_size: The chunk size

    Yields:
        A new iterator which returns the values of $iter in chunks
    """
    itr = iter(itr)
    for _ in itertools.count():
        chunk = []
        for _ in range(chunk_size):
            try:
                chunk.append(next(itr))
            except StopIteration:
                break  # finished

        yield chunk

        if len(chunk) < chunk_size:  # no more in iterator
            return


def parmap(  # noqa
    f: Callable[[Any], Any],
    X: Sized,  # pylint: disable
    nprocs: int = multiprocessing.cpu_count(),  # noqa
    force_parallel: bool = False,
    chunk_size: int = 1,
    use_tqdm: bool = False,
    keep_child_tqdm: bool = True,
    **tqdm_kwargs: object
) -> List[Any]:
    """Utility function for doing parallel calculations with multiprocessing.

    Splits the parameters into chunks (if wanted) and calls.
    Equivalent to list(map(func, params_iter))
    Args:
        f: The function we want to calculate for each element
        X: The parameters for the function (each element ins a list)
        chunk_size: Optional, the chunk size for the workers to work on
        nprocs: The number of procs to use (defaults for all cores)
        use_tqdm: Whether to use tqdm (default to False)
        force_parallel: force parmap to run in parallel (even if force_serial=True)
        keep_child_tqdm: Flag to indicate whether to keep children's processes tqdms active
        tqdm_kwargs: kwargs passed to tqdm

    Returns:
        The list of results after applying func to each element

    Has problems with using self.___ as variables in f (causes self to be pickled)
    """
    if len(X) == 0:
        return []  # like map
    if nprocs != multiprocessing.cpu_count() and len(X) < nprocs * chunk_size:
        chunk_size = 1  # use chunk_size = 1 if there is enough procs for a batch size of 1

    nprocs = int(max(1, min(nprocs, math.floor(len(X) / chunk_size))))  # at least 1
    if len(X) < nprocs:
        if nprocs != multiprocessing.cpu_count():
            print("parmap too much procs")
        nprocs = len(X)  # too much procs

    args = zip(X, [f] * len(X), range(len(X)), [keep_child_tqdm] * len(X))  # type: ignore
    if chunk_size > 1:
        args = list(chunk_iterator(args, chunk_size))
        s_fun = _chunk_spawn_fun  # type: ignore
    else:
        s_fun = _spawn_fun  # type: ignore

    if (nprocs == 1 and not force_parallel) or force_serial:  # we want it serial (maybe for profiling)
        return list(map(f, tqdm(X, disable=not use_tqdm, **tqdm_kwargs)))

    try:  # try-catch hides bugs
        global proc_count
        old_proc_count = proc_count
        proc_count = nprocs
        p = Pool(nprocs)
        p.restart(force=True)
        # can throw if current proc is daemon
        if use_tqdm:
            retval_par = tqdm(p.imap(s_fun, args), total=int(len(X) / chunk_size), **tqdm_kwargs)
        else:
            retval_par = p.map(s_fun, args)

        retval = list(retval_par)  # make it like the original map
        if chunk_size > 1:
            retval = flatten(retval)

        p.terminate()
        proc_count = old_proc_count
        global i
        i += 1
    except AssertionError:
        # if e == "daemonic processes are not allowed to have children":  # noqa
        retval = list(map(f, tqdm(X, disable=not use_tqdm, **tqdm_kwargs)))  # can't have pool inside pool
    return retval
