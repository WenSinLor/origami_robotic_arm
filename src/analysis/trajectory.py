import numpy as np
from pathlib import Path
from typing import Tuple

class TrajectoryAnalyzer:
    """
    Manages loading and post-processing of tracking data.
    Supports both .npz files and HDF5 (.h5 / .hdf5) files.

    HDF5 expected layout:
        /time_series/nodes/positions  (T, N, 3)
        /time_series/nodes/time       (T,)       [optional]
    """
    def __init__(self, data_path: str, fps: float = 60.0):
        self.data_path = Path(data_path)
        self.fps = fps
        self.trajectories = None  # Shape: (T, N, 3)
        self.timestamps = None
        self._load_data()

    def _load_data(self):
        """Internal method to load data on initialization."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")

        if self.data_path.suffix in ('.h5', '.hdf5'):
            self._load_hdf5()
        else:
            self._load_npz()

    def _load_npz(self):
        with np.load(self.data_path) as data:
            if 'trajectories' not in data:
                raise KeyError("The .npz file does not contain a 'trajectories' array.")
            self.trajectories = data['trajectories']

    def _load_hdf5(self):
        import h5py
        with h5py.File(self.data_path, 'r') as f:
            pos_path = 'time_series/nodes/positions'
            if pos_path not in f:
                raise KeyError(f"HDF5 file does not contain dataset '{pos_path}'.")
            self.trajectories = f[pos_path][:]

            time_path = 'time_series/nodes/time'
            if time_path in f:
                self.timestamps = f[time_path][:]
            
    def get_displacement(self, node_idx: int, axis_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates displacement relative to the first frame (t=0).
        
        Args:
            node_idx (int): Index of the marker (0-8)
            axis_idx (int): 0=X, 1=Y, 2=Z
            
        Returns:
            (time_axis, displacement_values)
        """
        # 1. Validation
        num_frames, num_nodes, _ = self.trajectories.shape
        if node_idx >= num_nodes:
            raise ValueError(f"Node index {node_idx} out of range (Max: {num_nodes-1})")

        # 2. Extract Data
        raw_coords = self.trajectories[:, node_idx, axis_idx]
        
        # 3. Calculate Relative Displacement
        # Displacement = Current_Pos - Initial_Pos
        initial_pos = raw_coords[0]
        displacement = raw_coords - initial_pos
        
        # 4. Generate Time Axis (use stored timestamps if available)
        if self.timestamps is not None:
            time_axis = self.timestamps
        else:
            time_axis = np.arange(num_frames) / self.fps
        
        return time_axis, displacement

    @property
    def node_count(self) -> int:
        return self.trajectories.shape[1] if self.trajectories is not None else 0
