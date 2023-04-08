import sys as _sys


def helloworld(comm, args=None, verbose=True):
    """Hello, World! using MPI."""
    # pylint: disable=import-outside-toplevel
    from argparse import ArgumentParser
    parser = ArgumentParser(prog=__name__ + " helloworld")
    parser.add_argument("-q", "--quiet", action="store_false",
                        dest="verbose", default=verbose)
    options = parser.parse_args(args)

    from . import MPI

    # comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    message = ("Hello, World! I am process %*d of %d on %s.\n"
               % (len(str(size - 1)), rank, size, name))
    comm.Barrier()
    if rank > 0:
        comm.Recv([None, 'B'], rank - 1)
    if options.verbose:
        _sys.stdout.write(message)
        _sys.stdout.flush()
    if rank < size - 1:
        comm.Send([None, 'B'], rank + 1)
    comm.Barrier()

    return message

if __name__ == "__name__":
    helloworld()