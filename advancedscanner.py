#!/usr/bin/env python3
"""
advancedscanner.py — Entry-point wrapper for the Advanced Scanner.

This thin wrapper lets you invoke the scanner from the repository root::

    python advancedscanner.py -H 192.168.1.1 -x web
    python advancedscanner.py -i
    python advancedscanner.py --list

All logic lives in security/advanced_scanner.py.

⚠️  LEGAL NOTICE: Only scan systems you own or have explicit written
    permission to test.  Unauthorised port scanning may violate computer-fraud
    laws in your jurisdiction.
"""

import sys
import os

# Make sure the repo root is on the path so the security package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from security.advanced_scanner import _cli

if __name__ == "__main__":
    _cli()
