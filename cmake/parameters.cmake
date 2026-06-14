# define default parameters

set_if_empty(NMASS 0)

# cuda options
if(CUDA)
  set(CUDA_OPTION "USE_CUDA")
else()
  set(CUDA_OPTION "NOT_USE_CUDA")
endif()

# ucx options
if(UCX)
  set(UCX_OPTION "USE_UCX")
else()
  set(UCX_OPTION "NOT_USE_UCX")
endif()

# netcdf options
if(NOT NETCDF OR NOT DEFINED NETCDF)
  set(NETCDF_OPTION "NO_NETCDFOUTPUT")
else()
  set(NETCDF_OPTION "NETCDFOUTPUT")
  find_package(NetCDF REQUIRED)
endif()

# pnetcdf options
if(NOT PNETCDF OR NOT DEFINED PNETCDF)
  set(PNETCDF_OPTION "NO_PNETCDFOUTPUT")
else()
  set(PNETCDF_OPTION "PNETCDFOUTPUT")
  find_package(PNetCDF REQUIRED)
endif()
