import os
import pytest
import xarray as xr
import numpy as np
from ccd import (
    get_variables,
    get_variable_periods,
    select_data,
    convert_data,
    define_periods,
    remove_repeated_variables,
    convert_data_monthly,
    convert_data_yearly,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
my_data_folder = os.path.join(THIS_DIR, 'data')
output_folder = os.path.join(THIS_DIR, 'output')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

def test_get_variables():
    # Test if the function returns a list of variables
    variables = get_variables(my_data_folder)
    assert isinstance(variables, list)
    assert len(variables) == 6
    assert "temp" in variables
    assert "a_ice" in variables

def test_get_variable_periods():
    # Test if the function returns a dictionary with start and end times for each variable
    variable_periods = get_variable_periods(["temp", "a_ice"], my_data_folder)
    assert isinstance(variable_periods, dict)
    assert len(variable_periods) == 2
    assert "temp" in variable_periods.keys()
    assert "a_ice" in variable_periods.keys()
    assert isinstance(variable_periods["temp"], dict)
    assert isinstance(variable_periods["a_ice"], dict)
    assert "start" in variable_periods["temp"].keys()
    assert "end" in variable_periods["temp"].keys()
    assert "start" in variable_periods["a_ice"].keys()
    assert "end" in variable_periods["a_ice"].keys()

def test_select_data():
    # Test if the function returns a subset of data within the specified time range
    data_selected = select_data("temp", "1948-01-01", "1949-12-31", my_data_folder)
    assert isinstance(data_selected, xr.Dataset)
    assert len(data_selected.time) == 2

def test_convert_data():
    # Test if the function converts data to netcdf format
    convert_data("temp", "1948-01-01", "1949-12-31", my_data_folder, output_folder, method="netcdf")
    assert os.path.exists(f"{output_folder}/temp.fesom.19481230_19491231.nc")
    os.remove(f"{output_folder}/temp.fesom.19481230_19491231.nc")
    convert_data("temp", "1948-01-01", "1949-12-31", my_data_folder, output_folder, method="netcdf_lz4")
    assert os.path.exists(f"{output_folder}/temp.fesom.19481230_19491231.nc")
    os.remove(f"{output_folder}/temp.fesom.19481230_19491231.nc")
    convert_data("temp", "1948-01-01", "1949-12-31", my_data_folder, output_folder, method="netcdf_zstd")
    assert os.path.exists(f"{output_folder}/temp.fesom.19481230_19491231.nc")
    os.remove(f"{output_folder}/temp.fesom.19481230_19491231.nc")

def test_define_periods():
    # Test if the function defines periods correctly
    month_chunks, year_chunks = define_periods("temp", "1948-01-01", "1949-12-31", my_data_folder, time_unit="monthly", num_items=1)
    assert isinstance(month_chunks, list)
    assert isinstance(year_chunks, list)
    assert len(month_chunks) == 2
    assert len(year_chunks) == 2
    assert len(month_chunks[0]) == 1
    assert len(year_chunks[0]) == 1

# def test_remove_repeated_variables():
#     # Test if the function removes repeated variables correctly
#     variables = ["temp", "a_ice"]
#     variable_periods = {"variable1": {"start": "2000-01-01", "end": "2000-12-31"}, "variable2": {"start": "2001-01-01", "end": "2001-12-31"}}
#     variables, variable_periods = remove_repeated_variables(variables, variable_periods, "test_data", "test_output")
#     assert len(variables) == 2
#     assert "variable1" in variables
#     assert "variable2" in variables
#     assert len(variable_periods) == 2
#     assert "variable1" in variable_periods.keys()
#     assert "variable2" in variable_periods.keys()

def test_convert_data_monthly():
    # Test if the function converts data to netcdf format for a given month and year
    convert_data_monthly("temp", [12], [1948], my_data_folder, output_folder, method="netcdf")
    assert os.path.exists(f"{output_folder}/temp.fesom.19481230_19481230.nc")
    os.remove(f"{output_folder}/temp.fesom.19481230_19481230.nc")

def test_convert_data_yearly():
    # Test if the function converts data to netcdf format for a given year
    convert_data_yearly("temp", [1948], my_data_folder, output_folder, method="netcdf")
    assert os.path.exists(f"{output_folder}/temp.fesom.19481230_19481230.nc")
    os.remove(f"{output_folder}/temp.fesom.19481230_19481230.nc")