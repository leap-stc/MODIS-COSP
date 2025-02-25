import os
from dataclasses import dataclass

import aiohttp
import apache_beam as beam
import fsspec
import numpy as np
import xarray as xr
from pangeo_forge_recipes.patterns import pattern_from_file_sequence
from pangeo_forge_recipes.storage import FSSpecTarget
from pangeo_forge_recipes.transforms import StoreToZarr

username, password = os.environ['EARTHDATA_USERNAME'], os.environ['EARTHDATA_PASSWORD']
client_kwargs = {
    'auth': aiohttp.BasicAuth(username, password),
    'trust_env': True,
}

# the urls are a bit hard to construct, so lets try with a few hardcoded ones
input_urls = [
    'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/62/MCD06COSP_M3_MODIS/2023/182/MCD06COSP_M3_MODIS.A2023182.062.2023223000656.nc',
    'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/62/MCD06COSP_M3_MODIS/2023/213/MCD06COSP_M3_MODIS.A2023213.062.2023254000930.nc',
    'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/62/MCD06COSP_M3_MODIS/2023/244/MCD06COSP_M3_MODIS.A2023244.062.2023285000449.nc',
]


# pattern = pattern_from_file_sequence(input_urls, concat_dim='time')
# pattern = pattern.prune(2)

# testing with local files for now
pattern = pattern_from_file_sequence(
    [
        '/Users/nrhagen/Documents/carbonplan/LEAP/feedstocks/MODIS-COSP/feedstock/MCD06COSP_M3_MODIS.A2023182.062.2023223000656.nc',
        '/Users/nrhagen/Documents/carbonplan/LEAP/feedstocks/MODIS-COSP/feedstock/MCD06COSP_M3_MODIS.A2023213.062.2023254000930.nc',
    ],
    concat_dim='time',
)


def _append_group_name_to_vars(dst: xr.DataTree) -> xr.DataTree:
    dataset_list = []
    for node in dst.children:
        time = np.datetime64(dst.attrs['time_coverage_start'])
        ds = dst[node].to_dataset()
        ds = ds.expand_dims(time=np.array([time]))

        group_name = dst[node].groups[0].split('/')[1]
        rename_dict = {f'{var}': f'{group_name}' + '_' + f'{var}' for var in list(ds)}
        ds = ds.rename(rename_dict)
        dataset_list.append(ds)

    return xr.merge(dataset_list)


@dataclass
class DatatreeToDataset(beam.PTransform):
    """Convert all datatree nodes into a single xarray dataset
    The netcdf file is organized into groups. We can open as a datatree, then parse all groups by
    adding the group name to the variable them, then merging back into a xarray dataset"""

    def _convert(self, dst: xr.DataTree) -> xr.Dataset:
        return _append_group_name_to_vars(dst)

    def expand(self, pcoll):
        return pcoll | '_convert' >> beam.MapTuple(lambda k, v: (k, self._convert(v)))


@dataclass
class OpenDatatreeXarray(beam.PTransform):
    """Open Xarray datatree"""

    def _open_dt(self, path: str) -> xr.DataTree:
        return xr.open_datatree(path)

    def expand(self, pcoll):
        return pcoll | '_open_dt' >> beam.MapTuple(lambda k, v: (k, self._open_dt(v)))


fs = fsspec.get_filesystem_class('file')()
target_root = FSSpecTarget(fs, 'modis_cosp')
with beam.Pipeline() as p:
    (
        p
        | beam.Create(pattern.items())
        # | OpenURLWithFSSpec(
        #     open_kwargs={'block_size': 0, 'client_kwargs': client_kwargs},
        #     max_concurrency=10,
        # )
        | OpenDatatreeXarray()
        | DatatreeToDataset()
        # | beam.Map(print)
        | StoreToZarr(
            target_root='.',
            store_name='MODIS_COSP.zarr',
            combine_dims=pattern.combine_dim_keys,
        )
    )
