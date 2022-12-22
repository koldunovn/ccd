import xarray as xr
import os
import sys
import glob
import xarray as xr
import numpy as np

from rich.console import Console
from rich.progress import track
from rich.traceback import install
import argparse
from tqdm import tqdm
from functools import cached_property
from rich.table import Table
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

install()
console = Console()


def get_list_size(path_list):
    if not isinstance(path_list, list):
        path_list = [path_list]

    total = 0
    for path in path_list:
        if os.path.isfile(path):
            total += os.path.getsize(path)
        else:
            total += get_list_size(path)
            # print(f'{path}: {result/1e+9}')
        # total += result
    return total


@dataclass
class Variable:
    name: str = ""
    nsteps: int = 0
    nlevels: int = 0
    tlevels: str = ""
    levels: np.array = np.array([])
    files: List[str] = None
    start: np.datetime64 = None
    end: np.datetime64 = None
    timefreq: np.timedelta64 = None
    dof_type: str = None
    dof: int = 0


class FileData:
    def __init__(
        self,
        folder,
        file_type="nc",
        naming_template="*.fesom.????",
        separator=".",
        variable_position=0,
    ):
        self.folder = folder
        self.file_type = file_type
        self.naming_template = naming_template
        self.separator = separator
        self.variable_position = variable_position
        self.files = self.get_files()
        self.variables = self.get_variables()
        self.get_files_for_variables()
        self.get_variable_properties()

    def get_files(self):
        all_files = glob.glob(self.folder + f"/{self.naming_template}.{self.file_type}")
        all_files.sort()
        return all_files

    def get_variables(self):
        varnames = []
        variables = {}
        for ffile in self.files:
            varname = os.path.basename(ffile).split(self.separator)[
                self.variable_position
            ]
            if varname not in varnames:
                varnames.append(varname)
                variables[varname] = Variable(varname)

        return variables

    def __getattr__(self, name):
        # If the attribute is not found in the object, try looking it up in the variables dictionary
        return self.variables[name]

    def get_files_for_variables(self):
        for variable in self.variables:
            self.variables[variable].files = []
            for ffile in self.files:
                varname = os.path.basename(ffile).split(self.separator)[
                    self.variable_position
                ]
                if varname == variable:
                    self.variables[variable].files.append(ffile)

    def get_variable_properties(self):
        pbar = tqdm(self.variables, leave=False, ncols=100, colour="green")
        for variable in pbar:
            pbar.set_description(f"Processing {variable:<11}")
            # variable_periods[variable] = {}
            data_in = xr.open_mfdataset(
                self.variables[variable].files, combine="by_coords"
            )
            self.variables[variable].start = data_in.time[0].values
            self.variables[variable].end = data_in.time[-1].values
            self.variables[variable].timefreq = (
                data_in.time[1].values - data_in.time[0].values
            )
            self.variables[variable].nsteps = len(data_in.time)

            if "nz" in data_in.dims:
                self.variables[variable].nlevels = len(data_in.nz)
                self.variables[variable].tlevels = "nz"
                self.variables[variable].levels = data_in.nz.data
            elif "nz1" in data_in.dims:
                self.variables[variable].nlevels = len(data_in.nz1)
                self.variables[variable].tlevels = "nz1"
                self.variables[variable].levels = data_in.nz1.data
            else:
                self.variables[variable].nlevels = 1
                self.variables[variable].tlevels = "2d"
                self.variables[variable].levels = np.array([0])

            if "elem" in data_in.dims:
                self.variables[variable].dof = len(data_in.elem)
                self.variables[variable].dof_type = "elem"
            elif "nod2" in data_in.dims:
                self.variables[variable].dof = len(data_in.nod2)
                self.variables[variable].dof_type = "nod2"

            data_in.close()
        # console.print(variable_periods)

    def get_variable_size(self, variable):
        return get_list_size(self.variables[variable].files)

    @cached_property
    def sizes(self):
        sizes = {}
        for variable in self.variables:
            sizes[variable] = self.get_variable_size(variable)
        self.sizes = sizes
        return sizes

    @cached_property
    def total_size(self):
        total = 0

        for variable in self.sizes:
            total += self.sizes[variable]

        total_size = total / 1e9
        self.total_size = total_size
        return total_size

    def describe(self):
        #
        table = Table(title=f"Variables in {self.folder} folder")
        table.add_column("Variable", justify="right", style="cyan", no_wrap=True)
        table.add_column("Size", style="magenta")
        table.add_column("Range", justify="right", style="green")
        table.add_column("Steps", justify="right", style="green")
        table.add_column("Output FQC", justify="right", style="green")
        table.add_column("Levels", justify="right", style="blue")
        table.add_column("DOF", justify="right", style="blue")
        for variable in self.variables:
            start_string = np.datetime_as_string(
                self.variables[variable].start, unit="m"
            )  # .replace("-", "")
            stop_string = np.datetime_as_string(
                self.variables[variable].end, unit="m"
            )  # .replace("-", "")
            table.add_row(
                f"{variable}",
                f"{self.sizes[variable]/1e9:.2f} GB",
                f"{start_string} - {stop_string}",
                f"{self.variables[variable].nsteps}",
                f"{self.variables[variable].timefreq.astype('timedelta64[h]').astype('int')/24:.3f} days",
                f"{self.variables[variable].nlevels}",
                f"{self.variables[variable].dof/1e6:.3f} M",
            )
            # console.print(f"{variable}: {self.sizes[variable]/1e9:.2f} GB")
        console.print(table)
        console.print(f"Total size: {self.total_size:.2f} GB")

    def __repr__(self) -> str:
        return f"ClimateDataFolder({self.folder})"


def describe():
    parser = argparse.ArgumentParser(
        prog="Describe folder",
        description="Describe climate data in the folder",
    )
    parser.add_argument(
        "folder", type=str, default="./", help="Path folder with input files"
    )
    # parser.set_defaults(func=describe)
    parser.add_argument("--file_type", "-t", type=str, default="nc", help="File type")
    parser.add_argument(
        "--naming_template",
        "-n",
        type=str,
        default="*.fesom.????",
        help="Naming template",
    )
    parser.add_argument(
        "--separator", "-s", type=str, default=".", help="Separator between variables"
    )
    parser.add_argument(
        "--variable_position",
        "-p",
        type=int,
        default=0,
        help="Position of variable in the file name",
    )

    args = parser.parse_args()
    data = FileData(
        args.folder,
        args.file_type,
        args.naming_template,
        args.separator,
        args.variable_position,
    )

    data.describe()


# ifolder = "/Users/nkolduno/PYTHON/DATA/LCORE/"
# data = FileData(ifolder)
# data.describe()
# print(data.total_size)

if __name__ == "__main__":
    describe()
