"""
Stub file for snapy.output module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch

# Output
class OutputOptions:
    """
    Output configuration options.

    This class manages output parameters.
    """

    def __init__(self) -> None:
        """Initialize OutputOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def fid(self) -> int:
        """Get file ID."""
        ...

    @overload
    def fid(self, value: int) -> "OutputOptions":
        """Set file ID."""
        ...

    @overload
    def dt(self) -> float:
        """Get output time interval."""
        ...

    @overload
    def dt(self, value: float) -> "OutputOptions":
        """Set output time interval."""
        ...

    @overload
    def output_sumx1(self) -> bool:
        """Get x1 sum output flag."""
        ...

    @overload
    def output_sumx1(self, value: bool) -> "OutputOptions":
        """Set x1 sum output flag."""
        ...

    @overload
    def output_sumx2(self) -> bool:
        """Get x2 sum output flag."""
        ...

    @overload
    def output_sumx2(self, value: bool) -> "OutputOptions":
        """Set x2 sum output flag."""
        ...

    @overload
    def output_sumx3(self) -> bool:
        """Get x3 sum output flag."""
        ...

    @overload
    def output_sumx3(self, value: bool) -> "OutputOptions":
        """Set x3 sum output flag."""
        ...

    @overload
    def include_ghost_zones(self) -> bool:
        """Get ghost zone inclusion flag."""
        ...

    @overload
    def include_ghost_zones(self, value: bool) -> "OutputOptions":
        """Set ghost zone inclusion flag."""
        ...

    @overload
    def cartesian_vector(self) -> bool:
        """Get Cartesian vector flag."""
        ...

    @overload
    def cartesian_vector(self, value: bool) -> "OutputOptions":
        """Set Cartesian vector flag."""
        ...

    @overload
    def double_precision(self) -> bool:
        """Get NC_DOUBLE netcdf output flag."""
        ...

    @overload
    def double_precision(self, value: bool) -> "OutputOptions":
        """Set NC_DOUBLE netcdf output flag (netcdf writer only)."""
        ...

    @overload
    def x1_slice(self) -> Optional[float]:
        """Get x1 slice position."""
        ...

    @overload
    def x1_slice(self, value: Optional[float]) -> "OutputOptions":
        """Set x1 slice position."""
        ...

    @overload
    def x2_slice(self) -> Optional[float]:
        """Get x2 slice position."""
        ...

    @overload
    def x2_slice(self, value: Optional[float]) -> "OutputOptions":
        """Set x2 slice position."""
        ...

    @overload
    def x3_slice(self) -> Optional[float]:
        """Get x3 slice position."""
        ...

    @overload
    def x3_slice(self, value: Optional[float]) -> "OutputOptions":
        """Set x3 slice position."""
        ...

    @overload
    def variables(self) -> List[str]:
        """Get list of output variables."""
        ...

    @overload
    def variables(self, value: List[str]) -> "OutputOptions":
        """Set list of output variables."""
        ...

    @overload
    def file_type(self) -> str:
        """Get output file type."""
        ...

    @overload
    def file_type(self, value: str) -> "OutputOptions":
        """Set output file type."""
        ...

    @overload
    def data_format(self) -> str:
        """Get data format."""
        ...

    @overload
    def data_format(self, value: str) -> "OutputOptions":
        """Set data format."""
        ...

class OutputType:
    """
    Output type base class.

    This class manages output file generation.
    """

    @overload
    def __init__(self) -> None:
        """Initialize OutputType with default values."""
        ...

    @overload
    def __init__(self, options: OutputOptions) -> None:
        """
        Initialize OutputType with options.

        Args:
            options: Output configuration options
        """
        ...

    def __repr__(self) -> str: ...

    file_number: int
    next_time: float

    def increment_file_number(self) -> int:
        """
        Increment and return the file number.

        Returns:
            New file number
        """
        ...

class NetcdfOutput(OutputType):
    """
    NetCDF output implementation.

    This class handles NetCDF file output.
    """

    def __init__(self, options: OutputOptions) -> None:
        """
        Initialize NetcdfOutput.

        Args:
            options: Output configuration options
        """
        ...

    def __repr__(self) -> str: ...

    def write_output_file(
        self,
        block,  # MeshBlock object
        vars: Dict[str, torch.Tensor],
        time: float,
        wtflag: int = 0
    ) -> None:
        """
        Write output file.

        Args:
            block: MeshBlock object
            vars: Dictionary of variable tensors
            time: Current simulation time
            wtflag: Write flag (default 0)
        """
        ...
