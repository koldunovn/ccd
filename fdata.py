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
        self.varaible_periods = self.get_variable_periods()

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
        # for variable in track(variables, console=console):
        for variable in tqdm(self.variables):
            variable_periods[variable] = {}
            data_in = xr.open_mfdataset(
                self.files_for_variables[variable], combine="by_coords"
            )
            variable_periods[variable]["start"] = data_in.time[0].values
            variable_periods[variable]["end"] = data_in.time[-1].values
            data_in.close()
        # console.print(variable_periods)
        return variable_periods

    def get_variable_size(self, variable):
        return get_list_size(self.files_for_variables[variable])

    @cached_property
    def sizes(self):
        sizes = {}
        for variable in tqdm(self.variables):
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
        table = Table(title="Variables in folder")
        table.add_column("Variable", justify="right", style="cyan", no_wrap=True)
        table.add_column("Size", style="magenta")
        table.add_column("Range", justify="right", style="green")
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
            )
            # console.print(f"{variable}: {self.sizes[variable]/1e9:.2f} GB")
        console.print(table)
        console.print(f"Total size: {self.total_size:.2f} GB")
        # return


ifolder = "/Users/nkolduno/PYTHON/DATA/LCORE/"
data = FileData(ifolder)
data.describe()
# print(data.total_size)
