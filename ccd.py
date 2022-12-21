import os
import sys
import glob
import xarray as xr
import zarr
from numcodecs import Blosc
from dask.diagnostics import ProgressBar
import numpy as np

from tqdm import tqdm
from rich.console import Console
from rich.progress import track
from rich.traceback import install
import argparse
from joblib import Parallel, delayed
import dask
from dask.distributed import Client, progress

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
        elif "blowup" in ffile:
            continue
        varname = os.path.basename(ffile).split(".")[0]
        if varname not in variables:
            variables.append(varname)
    return variables


def get_variable_periods(variables, idata):
    variable_periods = {}
    # for variable in track(variables, console=console):
    for variable in tqdm(variables):
        variable_periods[variable] = {}
        data_in = xr.open_mfdataset(f"{idata}/{variable}.*.nc", combine="by_coords")
        variable_periods[variable]["start"] = data_in.time[0].values
        variable_periods[variable]["end"] = data_in.time[-1].values
        data_in.close()
    # console.print(variable_periods)
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


def remove_repeated_variables(variables, variable_periods, idata, odata):
    variables_out = get_variables(odata)
    variables_out_periods = get_variable_periods(variables_out, odata)
    console.print(f"variables_out: {variables_out}")
    console.print(f"variables: {variables}")
    toremove = []
    for variable in variables:
        if variable in variables_out:
            if (
                variable_periods[variable]["start"]
                == variables_out_periods[variable]["start"]
                and variable_periods[variable]["end"]
                == variables_out_periods[variable]["end"]
            ):
                # console.print(f"Start and end times are the same, we will remove [bold]{variable}[/bold] from the list", style="red")
                toremove.append(variable)
            elif (
                variable_periods[variable]["start"]
                == variables_out_periods[variable]["start"]
                and variable_periods[variable]["end"]
                > variables_out_periods[variable]["end"]
            ):
                variable_periods[variable]["start"] = variables_out_periods[variable][
                    "end"
                ] + np.timedelta64(1, "D")
                console.print(f"We will update the start of {variable}")
                console.print(variable_periods[variable]["start"])
                console.print(variable_periods[variable]["end"])
            else:
                console.print(f"We will not remove {variable}")
        else:
            console.print("Variable not in output folder")

    console.print(f"variables fo be removed {toremove}")
    variables = [x for x in variables if x not in toremove]
    return variables, variable_periods


def convert_data_monthly(variable, month, year, idata, odata, method="netcdf"):
    start_mon = f"{year[0]}-{str(month[0]).zfill(2)}"
    stop_mon = f"{year[-1]}-{str(month[-1]).zfill(2)}"
    print(f"Variable: {variable}, start_mon: {start_mon} stop_mon: {stop_mon}")
    convert_data(variable, start_mon, stop_mon, idata, odata, method=method)
    return 1


# @dask.delayed
def convert_data_yearly(variable, year, idata, odata, method="netcdf"):
    start_year = f"{year[0]}"
    stop_year = f"{year[-1]}"
    print(f"Variable: {variable}, start_year: {start_year} stop_year: {stop_year}")
    convert_data(variable, start_year, stop_year, idata, odata, method=method)
    return 1


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
        "--variables",
        "-v",
        type=str,
        default=None,
        help='List of variables to be converted. If not specified, all variables in the input folder will be converted.\
        Example: -v "temp, salt, u, v, w"',
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="If specified, repeated variables will be converted again.",
    )

    parser.add_argument(
        "--parallelism",
        "-p",
        type=str,
        default="serial",
        help='Parallelism. Available options: "serial", "joblib", "dask".',
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help='Number of workers to be used. Only used if parallelism is "joblib" or "dask".',
    )

    args = parser.parse_args()
    idata = args.input
    odata = args.output
    num_items = args.num_items
    time_unit = args.time_unit
    parallelism = args.parallelism

    if parallelism == "dask":
        client = Client(threads_per_worker=1, n_workers=args.workers)

    if args.variables is None:
        variables = get_variables(idata)
    else:
        variables = [x.strip() for x in args.variables.split(",")]

    console.print(variables)
    variable_periods = get_variable_periods(variables, idata)

    if args.repeat is False:
        variables, variable_periods = remove_repeated_variables(
            variables, variable_periods, idata, odata
        )

    # for variable in track(
    #     variables[:], console=console, description="Converting files"
    # ):
    for variable in tqdm(variables[:]):
        start = variable_periods[variable]["start"]
        stop = variable_periods[variable]["end"]

        if time_unit == "monthly":
            months, years = define_periods(
                variable, start, stop, idata, time_unit=time_unit, num_items=num_items
            )
            if parallelism == "serial":
                for month, year in zip(months, years):
                    convert_data_monthly(
                        variable, month, year, idata, odata, method="netcdf"
                    )
            elif parallelism == "joblib":
                Parallel(n_jobs=args.workers)(
                    delayed(convert_data_monthly)(
                        variable, month, year, idata, odata, method="netcdf"
                    )
                    for month, year in zip(months, years)
                )
            elif parallelism == "dask":
                results = []
                for month, year in zip(months, years):
                    x = dask.delayed(convert_data_monthly)(
                        variable, month, year, idata, odata, method="netcdf"
                    )
                    results.append(x)
                dask.compute(*results)

        elif time_unit == "yearly":
            _, years = define_periods(
                variable, start, stop, idata, time_unit=time_unit, num_items=num_items
            )
            if parallelism == "serial":
                for year in years:
                    convert_data_yearly(variable, year, idata, odata, method="netcdf")
                # convert_data(
                #     variable, start_year, stop_year, idata, odata, method="netcdf"
                # )
            elif parallelism == "joblib":
                Parallel(n_jobs=args.workers)(
                    delayed(convert_data_yearly)(
                        variable, year, idata, odata, method="netcdf"
                    )
                    for year in years
                )
            elif parallelism == "dask":
                results = []
                for year in years:
                    x = dask.delayed(convert_data_yearly)(
                        variable, year, idata, odata, method="netcdf"
                    )
                    results.append(x)
                results = dask.compute(*results)
                print(results)

        elif time_unit == "whole":
            convert_data(variable, start, stop, idata, odata, method="netcdf")
        else:
            raise ValueError("Frequency not defined")

        console.log(
            f"Var: {variable}, start: {np.datetime_as_string(start, unit='D').replace('-', '')}, end: {np.datetime_as_string(stop, unit='D').replace('-', '')}",
            style="bold green",
        )

        if parallelism == "dask":
            client.close()


if __name__ == "__main__":
    # args = parser.parse_args()
    # args.func(args)
    ccd()
