#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2021 Satpy developers
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
"""Composite classes for the AHI instrument."""

import logging

from satpy.dataset import combine_metadata
from satpy.composites import GenericCompositor

from satpy.composites import enhance2dataset, add_bands
import xarray as xr

LOG = logging.getLogger(__name__)


class GreenCorrector(GenericCompositor):
    """Corrector of the AHI green band to compensate for the deficit of chlorophyll signal."""

    def __init__(self, *args, fractions=(0.85, 0.15), **kwargs):
        """Set default keyword argument values."""
        # XXX: Should this be 0.93 and 0.07
        self.fractions = fractions
        super(GreenCorrector, self).__init__(*args, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Boost vegetation effect thanks to NIR (0.8µm) band."""
        LOG.info('Boosting vegetation on green band')

        projectables = self.match_data_arrays(projectables)
        new_green = sum(fraction * value for fraction, value in zip(self.fractions, projectables))
        new_green.attrs = combine_metadata(*projectables)
        return super(GreenCorrector, self).__call__((new_green,), **attrs)


class BackgroundCompositor(GenericCompositor):

    def __init__(self, name, overlay=False, *args, **kwargs):

        self.overlay = overlay
        super(BackgroundCompositor, self).__init__(name, *args, **kwargs)

    def _overlay(self, background, foreground, alpha=None):
        if alpha is not None:
            base = background.where(alpha)
        else:
            base = background
        first_pass = xr.where(base >= 0.5, 1 - 2 * (1-background) * (1-foreground), base)
        data = xr.where(base < 0.5, 2 * background * foreground, first_pass)
        return data

    def __call__(self, projectables, *args, **kwargs):
        """Call the compositor."""
        projectables = self.match_data_arrays(projectables)

        # Get enhanced datasets
        foreground = enhance2dataset(projectables[0])
        background = enhance2dataset(projectables[1])

        # Adjust bands so that they match
        # L/RGB -> RGB/RGB
        # LA/RGB -> RGBA/RGBA
        # RGB/RGBA -> RGBA/RGBA
        foreground = add_bands(foreground, background['bands'])
        background = add_bands(background, foreground['bands'])

        # Get merged metadata
        attrs = combine_metadata(foreground, background)
        if attrs.get('sensor') is None:
            # sensor can be a set
            attrs['sensor'] = self._get_sensors(projectables)

        # Stack the images
        if 'A' in foreground.attrs['mode']:
            # Use alpha channel as weight and blend the two composites
            alpha = foreground.sel(bands='A')
            data = []
            # NOTE: there's no alpha band in the output image, it will
            # be added by the data writer
            for band in foreground.mode[:-1]:
                fg_band = foreground.sel(bands=band)
                bg_band = background.sel(bands=band)
                if self.overlay:
                    chan = self._overlay(bg_band, fg_band, alpha)
                else:
                    chan = (fg_band * alpha + bg_band * (1 - alpha))
                chan = xr.where(chan.isnull(), bg_band, chan)
                data.append(chan)
        else:
            if self.overlay:
                data = self._overlay(background, foreground)
            else:
                data = xr.where(foreground.isnull(), background, foreground)
            # Split to separate bands so the mode is correct
            data = [data.sel(bands=b) for b in data['bands']]

        res = super(BackgroundCompositor, self).__call__(data, **kwargs)
        res.attrs.update(attrs)
        return res