import os
import sys
import glob
import xarray as xr
import zarr
from numcodecs import Blosc
from dask.diagnostics import ProgressBar
import numpy as np

# from tqdm import tqdm
from rich.console import Console
from rich.progress import track
from rich.traceback import install
import argparse

install()

console = Console()
zarr_compressor = Blosc(cname="zstd", clevel=3)  # , shuffle=Blosc.AUTOSHUFFLE)
netcdf_compressor = {
    "zlib": True,
    "complevel": 1,
}


def get_variables(idata):
    files = glob.glob(f"{idata}/*.nc")
    files.sort()
    # print(files)
    variables = []
    for ffile in files:
        if "fesom.mesh.diag.nc" in ffile:
            continue
        varname = os.path.basename(ffile).split(".")[0]
        if varname not in variables:
            variables.append(varname)
    return variables


def get_variable_periods(variables, idata):
    variable_periods = {}
    for variable in track(variables, console=console):
        variable_periods[variable] = {}
        data_in = xr.open_mfdataset(f"{idata}/{variable}.*.nc", combine="by_coords")
        variable_periods[variable]["start"] = data_in.time[0].values
        variable_periods[variable]["end"] = data_in.time[-1].values
        data_in.close()
    console.print(variable_periods)
    return variable_periods


def select_data(variable, start, stop, idata):

    data2 = xr.open_mfdataset(
        f"{idata}/{variable}.*nc"
    )  # , chunks = variables[variable])
    data_selected = data2.sel(time=slice(start, stop))
    # console.log(log_locals=True)

    return data_selected


def convert_data(variable, start, stop, idata, odata, method="netcdf"):

    data_selected = select_data(variable, start, stop, idata)
    start_string = np.datetime_as_string(
        data_selected.time[0].values, unit="D"
    ).replace("-", "")
    stop_string = np.datetime_as_string(
        data_selected.time[-1].values, unit="D"
    ).replace("-", "")

    if method == "zarr":
        data_selected.to_zarr(
            f"{odata}/{variable}.fesom.{start_string}_{stop_string}.zarr",
            encoding={variable: {"compressor": zarr_compressor}},
        )
    elif method == "netcdf":
        data_selected.to_netcdf(
            f"{odata}/{variable}.fesom.{start_string}_{stop_string}.nc",
            encoding={variable: netcdf_compressor},
        )


def define_periods(variable, start, stop, idata, time_unit="monthly", num_items=1):

    data2 = xr.open_mfdataset(
        f"{idata}/{variable}.*nc"
    )  # , chunks = variables[variable])
    data_selected = data2.sel(time=slice(start, stop))

    if time_unit == "monthly":
        months = data_selected.time.dt.month.values
        years = data_selected.time.dt.year.values
        month_chunks = [
            months[x : x + num_items] for x in range(0, len(months), num_items)
        ]
        year_chunks = [
            years[x : x + num_items] for x in range(0, len(years), num_items)
        ]
    elif time_unit == "yearly":
        years = np.unique(data_selected.time.dt.year.values)
        year_chunks = [
            years[x : x + num_items] for x in range(0, len(years), num_items)
        ]
        months = np.ones_like(years)
        month_chunks = [
            months[x : x + num_items] for x in range(0, len(months), num_items)
        ]

    return month_chunks, year_chunks


def ccd(args=None):
    parser = argparse.ArgumentParser(
        prog="ccd (compress climate data)",
        description="Compresses climate data to netCDF or zarr format.",
    )
    parser.add_argument("input", help="Path folder with input files")
    parser.add_argument("output", help="Path folder with output files")
    parser.add_argument(
        "--time_unit",
        "-t",
        default="yearly",
        type=str,
        help="Time unit of the output files. Available options: 'monthly', 'yearly', 'whole'.",
    )

    parser.add_argument(
        "--num_items",
        "-n",
        default="1",
        type=int,
        help="Number of items in each file. For example, if time_unit is 'monthly' and num_items is 12, then each file will contain one year of data.",
    )

    parser.add_argument(
        "--box",
        "-b",
        type=str,
        default="-180.0, 180.0, -80.0, 90.0",
        help="Several options are available:\
            - Map boundaries in -180 180 -90 90 format that will be used for interpolation.\
            - Use one of the predefined regions. Available: gs (Golf Stream), \
                trop (Atlantic Tropics), arctic, gulf (also Golf Stream, but based on Mercator projection.)\))",
    )

    args = parser.parse_args()
    idata = args.input
    odata = args.output
    num_items = args.num_items
    time_unit = args.time_unit

    variables = get_variables(idata)
    variable_periods = get_variable_periods(variables, idata)

    for variable in track(
        variables[:], console=console, description="Converting files"
    ):
        start = variable_periods[variable]["start"]
        stop = variable_periods[variable]["end"]

        if time_unit == "monthly":
            months, years = define_periods(
                variable, start, stop, idata, time_unit=time_unit, num_items=num_items
            )
            for month, year in zip(months, years):
                start_mon = f"{year[0]}-{str(month[0]).zfill(2)}"
                stop_mon = f"{year[-1]}-{str(month[-1]).zfill(2)}"
                convert_data(
                    variable, start_mon, stop_mon, idata, odata, method="netcdf"
                )
        elif time_unit == "yearly":
            _, years = define_periods(
                variable, start, stop, idata, time_unit=time_unit, num_items=num_items
            )
            for year in years:
                start_year = f"{year[0]}"
                stop_year = f"{year[-1]}"
                convert_data(
                    variable, start_year, stop_year, idata, odata, method="netcdf"
                )
        elif time_unit == "whole":
            convert_data(variable, start, stop, idata, odata, method="netcdf")
        else:
            raise ValueError("Frequency not defined")

        console.log(
            f"Var: {variable}, start: {np.datetime_as_string(start, unit='D').replace('-', '')}, end: {np.datetime_as_string(stop, unit='D').replace('-', '')}",
            style="bold green",
        )


if __name__ == "__main__":
    # args = parser.parse_args()
    # args.func(args)
    ccd()
