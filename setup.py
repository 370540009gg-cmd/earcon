# -*- coding: utf-8 -*-
"""Compatibility shim for legacy pip / build frontends that don't fully
support PEP 660 pyproject-only builds from a git URL (produces
UNKNOWN-0.0.0 sdists). Real metadata lives in pyproject.toml.
"""

from setuptools import setup

setup()
