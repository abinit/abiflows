#!/usr/bin/env python
"""
Command line interface to the database.
"""
from __future__ import print_function, division, unicode_literals

import sys
import os
import argparse
import shlex
import time
import socket

from abiflows.core.server import AbiFlowsServer, daemonize_server, kill_server
from abiflows.core.client import AbiFlowsClient


def main():

    def str_examples():
        examples = """\
Usage example:\n

    abirun.py [FLOWDIR] rapid                    => Keep repeating, stop when no task can be executed.

    Options for developers:

        abirun.py prof ABIRUN_ARGS               => to profile abirun.py
"""
        return examples

    def show_examples_and_exit(err_msg=None, error_code=1):
        """Display the usage of the script."""
        sys.stderr.write(str_examples())
        if err_msg: sys.stderr.write("Fatal Error\n" + err_msg + "\n")
        sys.exit(error_code)

    # Parent parse for common options.
    copts_parser = argparse.ArgumentParser(add_help=False)

    copts_parser.add_argument('-v', '--verbose', default=0, action='count', # -vv --> verbose=2
                              help='verbose, can be supplied multiple times to increase verbosity')
    copts_parser.add_argument('--loglevel', default="ERROR", type=str,
                               help="set the loglevel. Possible values: CRITICAL, ERROR (default), WARNING, INFO, DEBUG")

    # Build the main parser.
    parser = argparse.ArgumentParser(epilog=str_examples(), formatter_class=argparse.RawDescriptionHelpFormatter)

    # Create the parsers for the sub-commands
    subparsers = parser.add_subparsers(dest='command', help='sub-command help', description="Valid subcommands")

    subparsers.add_parser('version', parents=[copts_parser], help='Show version number and exit')

    # Subparser for connect.
    p_start = subparsers.add_parser('start', parents=[copts_parser], help="Start the server.")
    p_start.add_argument("-r", '--remove-pid', default=False, action="store_true", 
                         help="Remove the pid file of the server before starting new instance.")

    p_shutdown = subparsers.add_parser('shutdown', parents=[copts_parser], help="Shut down the server.")

    # Subparser for connect.
    p_connect = subparsers.add_parser('connect', parents=[copts_parser], help="Test connection.")

    # Subparser for connect.
    p_kill = subparsers.add_parser('kill', parents=[copts_parser], help="Kill server.")

    # Parse command line.
    try:
        options = parser.parse_args()
    except Exception as exc: 
        show_examples_and_exit(error_code=1)

    # loglevel is bound to the string value obtained from the command line argument. 
    # Convert to upper case to allow the user to specify --loglevel=DEBUG or --loglevel=debug
    import logging
    numeric_level = getattr(logging, options.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % options.loglevel)
    logging.basicConfig(level=numeric_level)

    retcode = 0

    if options.command == "version":
        from abiflows.core.release import version
        print(version)
        return 0

    elif options.command == "start":
        daemonize_server(options.remove_pid)

    elif options.command == "shutdown":
        with AbiFlowsClient() as client:
            client.shutdown_server()

    elif options.command == "kill":
        retcode = kill_server()

    #elif options.command == "restart":
    #    server = AbiFlowsServer()

    #elif options.command == "status":
    #elif options.command == "info":

    elif options.command == "connect":
        with AbiFlowsClient() as client:
            client.send_flow("hello flow")

    else:
        raise RuntimeError("Don't know what to do with command %s!" % options.command)

    return retcode


if __name__ == "__main__":
    retcode = 0
    do_prof, do_tracemalloc = 2* [False]
    try:
        do_prof = sys.argv[1] == "prof"
        do_tracemalloc = sys.argv[1] == "tracemalloc"
        if do_prof or do_tracemalloc: sys.argv.pop(1)
    except: 
        pass

    if do_prof:
        import pstats, cProfile
        cProfile.runctx("main()", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()

    elif do_tracemalloc:
        # Requires py3.4
        import tracemalloc
        tracemalloc.start()

        retcode = main()

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        print("[Top 10]")
        for stat in top_stats[:10]:
            print(stat)
    else:
        sys.exit(main())

    #open_hook.print_open_files()
    sys.exit(retcode)
