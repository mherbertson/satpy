#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2019 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Satpy Configuration directory and file handling."""

import glob
import logging
import os
import sys
from collections import OrderedDict

import pkg_resources
import yaml
from yaml import BaseLoader
from donfig import Config
import appdirs

try:
    from yaml import UnsafeLoader
except ImportError:
    from yaml import Loader as UnsafeLoader

LOG = logging.getLogger(__name__)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
# FIXME: Use package_resources?
PACKAGE_CONFIG_PATH = os.path.join(BASE_PATH, 'etc')

_satpy_dirs = appdirs.AppDirs(appname='satpy', appauthor='pytroll')
_CONFIG_DEFAULTS = {
    'cache_dir': _satpy_dirs.user_cache_dir,
    'data_dir': _satpy_dirs.user_data_dir,
    'config_path': [],
}

# Satpy main configuration object
# See https://donfig.readthedocs.io/en/latest/configuration.html
# for more information.
#
# Configuration values will be loaded from files at:
# 1. The builtin package satpy.yaml (not present currently)
# 2. $SATPY_ROOT_CONFIG (default: /etc/satpy/satpy.yaml)
# 3. <python-env-prefix>/etc/satpy/satpy.yaml
# 4. ~/.config/satpy/satpy.yaml
# 5. ~/.satpy/satpy.yaml
# 6. $SATPY_CONFIG_PATH/satpy.yaml if present (colon separated)
_CONFIG_PATHS = [
    os.path.join(PACKAGE_CONFIG_PATH, 'satpy.yaml'),
    os.getenv('SATPY_ROOT_CONFIG', os.path.join('/etc', 'satpy', 'satpy.yaml')),
    os.path.join(sys.prefix, 'etc', 'satpy', 'satpy.yaml'),
    os.path.join(os.path.expanduser('~'), '.config', 'satpy', 'satpy.yaml'),
    os.path.join(os.path.expanduser('~'), '.satpy', 'satpy.yaml'),
]
# The above files can also be directories. If directories all files
# with `.yaml`., `.yml`, or `.json` extensions will be used.

_ppp_config_dir = os.getenv('PPP_CONFIG_DIR', None)
_satpy_config_path = os.getenv('SATPY_CONFIG_PATH', None)
if _ppp_config_dir is not None and _satpy_config_path is None:
    LOG.warning("'PPP_CONFIG_DIR' is deprecated. Please use 'SATPY_CONFIG_PATH' instead.")
    _satpy_config_path = _ppp_config_dir
    os.environ['SATPY_CONFIG_PATH'] = _satpy_config_path

if _satpy_config_path is not None:
    # colon-separated are ordered by builtins -> custom
    # i.e. first/lowest priority to last/highest priority
    _satpy_config_path = _satpy_config_path.split(':')
    os.environ['SATPY_CONFIG_PATH'] = _satpy_config_path
    for config_dir in _satpy_config_path[::-1]:
        _CONFIG_PATHS.append(os.path.join(config_dir, 'satpy.yaml'))

_ancpath = os.getenv('SATPY_ANCPATH', None)
_data_dir = os.getenv('SATPY_DATA_DIR', None)
if _ancpath is not None and _data_dir is None:
    LOG.warning("'SATPY_ANCPATH' is deprecated. Please use 'SATPY_DATA_DIR' instead.")
    os.environ['SATPY_DATA_DIR'] = _ancpath

config = Config("satpy", defaults=[_CONFIG_DEFAULTS], paths=_CONFIG_PATHS)


def get_entry_points_config_dirs(name, include_config_path=True):
    """Get the config directories for all entry points of given name."""
    dirs = []
    for entry_point in pkg_resources.iter_entry_points(name):
        package_name = entry_point.module_name.split('.', 1)[0]
        new_dir = os.path.join(entry_point.dist.module_path, package_name, 'etc')
        if not dirs or dirs[-1] != new_dir:
            dirs.append(new_dir)
    if include_config_path:
        dirs.extend(config.get('config_path')[::-1])
    return dirs


def config_search_paths(filename, search_dirs=None, **kwargs):
    """Get series of configuration base paths where Satpy configs are located."""
    if search_dirs is None:
        search_dirs = config.get('config_path')[::-1]

    paths = [filename, os.path.basename(filename)]
    paths += [os.path.join(search_dir, filename) for search_dir in search_dirs]
    paths += [os.path.join(PACKAGE_CONFIG_PATH, filename)]
    paths = [os.path.abspath(path) for path in paths]

    if kwargs.get("check_exists", True):
        paths = [x for x in paths if os.path.isfile(x)]

    paths = list(OrderedDict.fromkeys(paths))
    # flip the order of the list so builtins are loaded first
    return paths[::-1]


def glob_config(pattern, search_dirs=None):
    """Return glob results for all possible configuration locations.

    Note: This method does not check the configuration "base" directory if the pattern includes a subdirectory.
          This is done for performance since this is usually used to find *all* configs for a certain component.
    """
    patterns = config_search_paths(pattern, search_dirs=search_dirs, check_exists=False)

    for pattern in patterns:
        for path in glob.iglob(pattern):
            yield path


def get_config_path(filename, *search_dirs):
    """Get the appropriate path for a filename, in that order: filename, ., PPP_CONFIG_DIR, package's etc dir."""
    paths = config_search_paths(filename, *search_dirs)

    for path in paths[::-1]:
        if os.path.exists(path):
            return path


def _check_yaml_configs(configs, key):
    """Get a diagnostic for the yaml *configs*.

    *key* is the section to look for to get a name for the config at hand.
    """
    diagnostic = {}
    for i in configs:
        for fname in i:
            with open(fname, 'r', encoding='utf-8') as stream:
                try:
                    res = yaml.load(stream, Loader=UnsafeLoader)
                    msg = 'ok'
                except yaml.YAMLError as err:
                    stream.seek(0)
                    res = yaml.load(stream, Loader=BaseLoader)
                    if err.context == 'while constructing a Python object':
                        msg = err.problem
                    else:
                        msg = 'error'
                finally:
                    try:
                        diagnostic[res[key]['name']] = msg
                    except (KeyError, TypeError):
                        # this object doesn't have a 'name'
                        pass
    return diagnostic


def _check_import(module_names):
    """Import the specified modules and provide status."""
    diagnostics = {}
    for module_name in module_names:
        try:
            __import__(module_name)
            res = 'ok'
        except ImportError as err:
            res = str(err)
        diagnostics[module_name] = res
    return diagnostics


def check_satpy(readers=None, writers=None, extras=None):
    """Check the satpy readers and writers for correct installation.

    Args:
        readers (list or None): Limit readers checked to those specified
        writers (list or None): Limit writers checked to those specified
        extras (list or None): Limit extras checked to those specified

    Returns: bool
        True if all specified features were successfully loaded.

    """
    from satpy.readers import configs_for_reader
    from satpy.writers import configs_for_writer

    print('Readers')
    print('=======')
    for reader, res in sorted(_check_yaml_configs(configs_for_reader(reader=readers), 'reader').items()):
        print(reader + ': ', res)
    print()

    print('Writers')
    print('=======')
    for writer, res in sorted(_check_yaml_configs(configs_for_writer(writer=writers), 'writer').items()):
        print(writer + ': ', res)
    print()

    print('Extras')
    print('======')
    module_names = extras if extras is not None else ('cartopy', 'geoviews')
    for module_name, res in sorted(_check_import(module_names).items()):
        print(module_name + ': ', res)
    print()
