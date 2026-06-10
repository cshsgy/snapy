# define default parameters

set_if_empty(NMASS 0)

# nccl options
if(NOT CUDA OR NOT DEFINED CUDA)
  set(NCCL_OPTION "NOT_USE_C10D_NCCL")
else()
  set(NCCL_OPTION "USE_C10D_NCCL")
endif()

# ucx options: the tag-matching backend is provided by the commux package
# (cmake/commux.cmake), which owns UCX discovery/bundling.
if(NOT UCX OR NOT DEFINED UCX)
  set(UCX_OPTION "NOT_USE_C10D_UCX")
else()
  set(UCX_OPTION "USE_C10D_UCX")
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
