import argparse
from logcreator.logcreator import Logcreator
"""
Handles arguments provided in comand line
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import datetime
import os
import time
import numpy as np

def get_args():
    """
    Returns list of args
    """
    return args


def parse_args(parser):
    """
    Read and parse args from comandline and store in args
    """
    if parser:
        global args
        args = parser.parse_args()
    else:
        raise EnvironmentError(
            Logcreator.info("Parsing of comand line parameters failed")
        )
    return args