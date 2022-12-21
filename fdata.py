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

install()
console = Console()

ifolder = "/Users/nkolduno/PYTHON/DATA/"

# def get_dir_size(start_path = '.'):
#     '''
#     https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
#     '''
#     total_size = 0
#     for dirpath, dirnames, filenames in os.walk(start_path):
#         for f in filenames:
#             fp = os.path.join(dirpath, f)
#             # skip if it is symbolic link
#             if not os.path.islink(fp):
#                 total_size += os.path.getsize(fp)

#     return total_size


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
        self.files_for_variables = self.get_files_for_variables()
        (
            self.varaible_periods,
            self.variable_steps,
            self.variable_nsteps,
            self.variable_vlevels,
            self.variable_dof,
        ) = self.get_variable_periods()

    def get_files(self):
        all_files = glob.glob(self.folder + f"/{self.naming_template}.{self.file_type}")
        all_files.sort()
        return all_files

    def get_variables(self):
        variables = []
        for ffile in self.files:
            varname = os.path.basename(ffile).split(self.separator)[
                self.variable_position
            ]
            if varname not in variables:
                variables.append(varname)

        return variables

    def get_files_for_variables(self):
        files_for_variables = {}
        for variable in self.variables:
            files_for_variables[variable] = []
            for ffile in self.files:
                varname = os.path.basename(ffile).split(self.separator)[
                    self.variable_position
                ]
                if varname == variable:
                    files_for_variables[variable].append(ffile)
        return files_for_variables

    def get_variable_periods(self):
        variable_periods = {}
        variable_steps = {}
        variable_nsteps = {}
        variable_vlevels = {}
        variable_dof = {}
        # for variable in track(variables, console=console):
        pbar = tqdm(self.variables, leave=False, ncols=100, colour="green")
        for variable in pbar:
            pbar.set_description("Processing %s" % variable)
            variable_periods[variable] = {}
            data_in = xr.open_mfdataset(
                self.files_for_variables[variable], combine="by_coords"
            )
            variable_periods[variable]["start"] = data_in.time[0].values
            variable_periods[variable]["end"] = data_in.time[-1].values
            variable_steps[variable] = data_in.time[1].values - data_in.time[0].values
            variable_nsteps[variable] = len(data_in.time)

            if "nz" in data_in.dims:
                variable_vlevels[variable] = len(data_in.nz)
            elif "nz1" in data_in.dims:
                variable_vlevels[variable] = len(data_in.nz1)
            else:
                variable_vlevels[variable] = 1

            if "elem" in data_in.dims:
                variable_dof[variable] = len(data_in.elem)
            elif "nod2" in data_in.dims:
                variable_dof[variable] = len(data_in.nod2)
            else:
                variable_dof[variable] = 0

            data_in.close()
        # console.print(variable_periods)
        return (
            variable_periods,
            variable_steps,
            variable_nsteps,
            variable_vlevels,
            variable_dof,
        )

    def get_variable_size(self, variable):
        return get_list_size(self.files_for_variables[variable])

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
                self.varaible_periods[variable]["start"], unit="m"
            )  # .replace("-", "")
            stop_string = np.datetime_as_string(
                self.varaible_periods[variable]["end"], unit="m"
            )  # .replace("-", "")
            table.add_row(
                f"{variable}",
                f"{self.sizes[variable]/1e9:.2f} GB",
                f"{start_string} - {stop_string}",
                f"{self.variable_nsteps[variable]}",
                f"{self.variable_steps[variable].astype('timedelta64[h]').astype('int')/24:.3f} days",
                f"{self.variable_vlevels[variable]}",
                f"{self.variable_dof[variable]/1e6:.3f} M",
            )
            # console.print(f"{variable}: {self.sizes[variable]/1e9:.2f} GB")
        console.print(table)
        console.print(f"Total size: {self.total_size:.2f} GB")
        # return


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
    # args = parser.parse_args()
    # args.func(args)
    describe()
