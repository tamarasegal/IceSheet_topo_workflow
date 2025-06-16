#!/usr/bin/env python3
"""
bin_to_cube_2ndgen.py - Complete Python conversion of bin_to_cube.F90

This program reads lat-lon terrain dataset from NetCDF file and bins it to an 
approximately 3km cubed-sphere grid and outputs the data in netCDF format.

The LANDM_COSLAT field is read in from a separate netCDF file and linearly
interpolated to the 3km cubed-sphere grid.

Author: Converted from Peter Hjort Lauritzen's Fortran code
Original: Nov 7, 2011
Python conversion: 2025
"""

import numpy as np
import netCDF4 as nc
import math
import sys
import os
import configparser
from datetime import datetime
from typing import Tuple, Optional

# Constants
PI = math.pi
PIQ = 0.25 * PI
RAD2DEG = 180.0 / PI
DEG2RAD = PI / 180.0
TINY = 1.0e-10
ROTATE_CUBE = 0.0

class TerrainProcessor:
    """Main class for terrain processing to cubed sphere grid"""
    
    def __init__(self, config_file: str = "bin_to_cube.ini"):
        """Initialize with configuration file"""
        self.config = self._read_config(config_file)
        self.ncube = self.config['ncube']
        self.raw_latlon_data_file = self.config['raw_latlon_data_file']
        self.output_file = self.config['output_file']
        self.landm_coslat_file = self.config.get('landm_coslat_file', 'landm_coslat.nc')
        
        print(f"Intermediate cubed-sphere resolution: {self.ncube}")
        
        # Initialize data arrays - will be set during processing
        self.im = None
        self.jm = None
        self.terr = None
        self.landfrac = None
        self.lon = None
        self.lat = None
        self.landm_coslat = None
        self.lon_landm = None
        self.lat_landm = None
        self.im_landm = None
        self.jm_landm = None
    
    def _read_config(self, config_file: str) -> dict:
        """Read configuration from INI file (replacing Fortran namelist)"""
        if not os.path.exists(config_file):
            # Create default config if it doesn't exist
            self._create_default_config(config_file)
        
        config = configparser.ConfigParser()
        config.read(config_file)
        
        try:
            return {
                'raw_latlon_data_file': config.get('binparams', 'raw_latlon_data_file'),
                'output_file': config.get('binparams', 'output_file'),
                'ncube': config.getint('binparams', 'ncube'),
                'landm_coslat_file': config.get('binparams', 'landm_coslat_file', fallback='landm_coslat.nc')
            }
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
            print(f"Error reading config file {config_file}: {e}")
            sys.exit(1)
    
    def _create_default_config(self, config_file: str):
        """Create a default configuration file"""
        config = configparser.ConfigParser()
        config['binparams'] = {
            'raw_latlon_data_file': 'input_terrain.nc',
            'output_file': 'output_cubed_sphere.nc',
            'ncube': '3000',
            'landm_coslat_file': 'landm_coslat.nc'
        }
        
        with open(config_file, 'w') as f:
            config.write(f)
        
        print(f"Created default config file: {config_file}")
        print("Please edit the file with your actual input/output filenames")
    
    def _validate_netcdf_file(self, filename: str, required_vars: list) -> bool:
        """Validate that NetCDF file exists and has required variables"""
        if not os.path.exists(filename):
            print(f"Error: NetCDF file does not exist: {filename}")
            return False
        
        try:
            with nc.Dataset(filename, 'r') as ncfile:
                missing_vars = [var for var in required_vars if var not in ncfile.variables]
                if missing_vars:
                    print(f"Error: Missing variables in {filename}: {missing_vars}")
                    return False
                
                # Check for required dimensions
                if 'lat' not in ncfile.dimensions or 'lon' not in ncfile.dimensions:
                    print(f"Error: Missing required dimensions (lat, lon) in {filename}")
                    return False
                    
        except Exception as e:
            print(f"Error reading NetCDF file {filename}: {e}")
            return False
        
        return True
    
    def read_terrain_data(self):
        """Read terrain data from NetCDF file with comprehensive error checking"""
        required_vars = ['landfract', 'htopo', 'lon', 'lat']
        
        if not self._validate_netcdf_file(self.raw_latlon_data_file, required_vars):
            sys.exit(1)
        
        print(f"Opening: {self.raw_latlon_data_file}")
        
        try:
            with nc.Dataset(self.raw_latlon_data_file, 'r') as ncfile:
                # Get dimensions
                self.im = len(ncfile.dimensions['lon'])
                self.jm = len(ncfile.dimensions['lat'])
                print(f"lon-lat dimensions: {self.im}, {self.jm}")
                
                if self.im < 2 or self.jm < 2:
                    raise ValueError(f"Invalid grid dimensions: {self.im} x {self.jm}")
                
                # Read data with proper data type handling
                try:
                    self.landfrac = ncfile.variables['landfract'][:].astype(np.float64)
                    self.terr = ncfile.variables['htopo'][:].astype(np.float64)
                    self.lon = ncfile.variables['lon'][:].astype(np.float64)
                    self.lat = ncfile.variables['lat'][:].astype(np.float64)
                except Exception as e:
                    raise ValueError(f"Error reading data arrays: {e}")
                
                # Validate data shapes
                if self.landfrac.shape != (self.jm, self.im):
                    raise ValueError(f"landfract shape mismatch: expected ({self.jm}, {self.im}), got {self.landfrac.shape}")
                if self.terr.shape != (self.jm, self.im):
                    raise ValueError(f"htopo shape mismatch: expected ({self.jm}, {self.im}), got {self.terr.shape}")
                if len(self.lon) != self.im:
                    raise ValueError(f"lon length mismatch: expected {self.im}, got {len(self.lon)}")
                if len(self.lat) != self.jm:
                    raise ValueError(f"lat length mismatch: expected {self.jm}, got {len(self.lat)}")
                
                # Validate coordinate ranges
                if np.any(self.lon < -180) or np.any(self.lon > 360):
                    print("Warning: longitude values outside expected range [-180, 360]")
                if np.any(self.lat < -90) or np.any(self.lat > 90):
                    raise ValueError("Latitude values outside valid range [-90, 90]")
                
                # Handle fill values and invalid data
                fill_value = getattr(ncfile.variables['htopo'], '_FillValue', None)
                if fill_value is not None:
                    self.terr = np.where(self.terr == fill_value, -9999, self.terr)
                
                fill_value = getattr(ncfile.variables['landfract'], '_FillValue', None)
                if fill_value is not None:
                    self.landfrac = np.where(self.landfrac == fill_value, -99.0, self.landfrac)
                
                print(f"min/max of 30sec land fraction: {np.min(self.landfrac):.3f}, {np.max(self.landfrac):.3f}")
                print(f"min/max of terrain elevation: {np.min(self.terr):.1f}, {np.max(self.terr):.1f}")
                
        except Exception as e:
            print(f"Fatal error reading terrain data: {e}")
            sys.exit(1)
        
        print("Done reading terrain data from netCDF file")
    
    def read_landm_coslat_data(self):
        """Read LANDM_COSLAT data with error checking"""
        required_vars = ['LANDM_COSLAT', 'lon', 'lat']
        
        if not self._validate_netcdf_file(self.landm_coslat_file, required_vars):
            sys.exit(1)
        
        print(f"Reading LANDM_COSLAT from file: {self.landm_coslat_file}")
        
        try:
            with nc.Dataset(self.landm_coslat_file, 'r') as ncfile:
                # Get dimensions
                self.im_landm = len(ncfile.dimensions['lon'])
                self.jm_landm = len(ncfile.dimensions['lat'])
                print(f"LANDM_COSLAT lon-lat dimensions: {self.im_landm}, {self.jm_landm}")
                
                if self.im_landm < 2 or self.jm_landm < 2:
                    raise ValueError(f"Invalid LANDM_COSLAT grid dimensions: {self.im_landm} x {self.jm_landm}")
                
                # Read data
                self.landm_coslat = ncfile.variables['LANDM_COSLAT'][:].astype(np.float64)
                self.lon_landm = ncfile.variables['lon'][:].astype(np.float64)
                self.lat_landm = ncfile.variables['lat'][:].astype(np.float64)
                
                # Validate shapes
                if self.landm_coslat.shape != (self.jm_landm, self.im_landm):
                    raise ValueError(f"LANDM_COSLAT shape mismatch: expected ({self.jm_landm}, {self.im_landm}), got {self.landm_coslat.shape}")
                
                # Handle fill values
                fill_value = getattr(ncfile.variables['LANDM_COSLAT'], '_FillValue', None)
                if fill_value is not None:
                    self.landm_coslat = np.where(self.landm_coslat == fill_value, -999999.99, self.landm_coslat)
                
                print(f"min/max of landm_coslat: {np.min(self.landm_coslat):.6f}, {np.max(self.landm_coslat):.6f}")
                
        except Exception as e:
            print(f"Fatal error reading LANDM_COSLAT data: {e}")
            sys.exit(1)
        
        print("Done reading LANDM_COSLAT data from netCDF file")
    
    def compute_raw_volume(self) -> float:
        """Compute volume for raw lat-lon data"""
        print("Computing volume for raw data")
        
        if self.lon is None or self.lat is None or self.terr is None:
            raise RuntimeError("Terrain data not loaded")
        
        vol = 0.0
        dx = self.lon[1] - self.lon[0]
        dx_rad = dx * DEG2RAD
        area_latlon = 0.0
        
        # Check for uniform longitude spacing
        lon_diffs = np.diff(self.lon)
        if not np.allclose(lon_diffs, lon_diffs[0], rtol=1e-6):
            print("Warning: non-uniform longitude spacing detected")
        
        for j in range(self.jm):
            for i in range(self.im):
                # More robust area calculation
                lat_south = -90.0 + dx * (j - 1) if j > 0 else -90.0
                lat_north = -90.0 + dx * j
                
                darea_latlon = dx_rad * (math.sin(DEG2RAD * lat_north) - 
                                       math.sin(DEG2RAD * lat_south))
                
                if not math.isnan(self.terr[j, i]) and self.terr[j, i] != -9999:
                    vol += float(self.terr[j, i]) * darea_latlon
                
                area_latlon += darea_latlon
        
        if area_latlon == 0:
            raise RuntimeError("Total area is zero - check input data")
        
        vol = vol / area_latlon
        
        print(f"Consistency of lat-lon area: {area_latlon - 4.0*PI:.6e}")
        print(f"Mean elevation (raw data): {vol:.2f} m")
        
        return vol
    
    def cubed_sphere_abp_from_rll(self, lon: float, lat: float) -> Tuple[float, float, int]:
        """
        Convert lat-lon to cubed sphere (alpha, beta, panel) coordinates
        Returns alpha, beta (in radians), panel (1-6)
        """
        # Convert to Cartesian coordinates
        xx = math.cos(lon - ROTATE_CUBE) * math.cos(lat)
        yy = math.sin(lon - ROTATE_CUBE) * math.cos(lat)
        zz = math.sin(lat)
        
        pm = max(abs(xx), abs(yy), abs(zz))
        
        if pm == 0:
            raise ValueError(f"Invalid coordinate conversion: lon={lon}, lat={lat}")
        
        # Check which coordinate has maximum absolute value
        ix = 1 if pm == abs(xx) and xx > 0 else (-1 if pm == abs(xx) and xx < 0 else 0)
        iy = 1 if pm == abs(yy) and yy > 0 else (-1 if pm == abs(yy) and yy < 0 else 0)
        iz = 1 if pm == abs(zz) and zz > 0 else (-1 if pm == abs(zz) and zz < 0 else 0)
        
        # Panel assignments (following original Fortran exactly)
        if iz == 1:
            ipanel = 6
            sx, sy, sz = yy, -xx, zz
        elif iz == -1:
            ipanel = 5
            sx, sy, sz = yy, xx, -zz
        elif ix == 1 and iy != 1:
            ipanel = 1
            sx, sy, sz = yy, zz, xx
        elif ix == -1 and iy != -1:
            ipanel = 3
            sx, sy, sz = -yy, zz, -xx
        elif iy == 1 and ix != -1:
            ipanel = 2
            sx, sy, sz = -xx, zz, yy
        elif iy == -1 and ix != 1:
            ipanel = 4
            sx, sy, sz = xx, zz, -yy
        else:
            raise ValueError(f"CubedSphereABPFromRLL failed: (xx,yy,zz)=({xx},{yy},{zz}), "
                           f"pm={pm}, (ix,iy,iz)=({ix},{iy},{iz})")
        
        if abs(sz) < TINY:
            raise ValueError(f"sz too small in coordinate conversion: {sz}")
        
        alpha = math.atan(sx / sz)
        beta = math.atan(sy / sz)
        
        return alpha, beta, ipanel
    
    def cubed_sphere_xyz_from_abp(self, alpha: float, beta: float, ipanel: int) -> Tuple[float, float, float]:
        """Convert cubed sphere coordinates to Cartesian coordinates"""
        if not (1 <= ipanel <= 6):
            raise ValueError(f"Panel out of range: {ipanel}")
        
        a1 = math.tan(alpha)
        b1 = math.tan(beta)
        
        denom = 1.0 + a1*a1 + b1*b1
        if denom <= 0:
            raise ValueError(f"Invalid denominator in coordinate conversion: {denom}")
        
        sz = denom**(-0.5)
        sx = sz * a1
        sy = sz * b1
        
        # Panel assignments (must match the forward transformation exactly)
        if ipanel == 6:
            xx, yy, zz = -sy, sx, sz
        elif ipanel == 5:
            xx, yy, zz = sy, sx, -sz
        elif ipanel == 1:
            xx, yy, zz = sz, sx, sy
        elif ipanel == 3:
            xx, yy, zz = -sz, -sx, sy
        elif ipanel == 2:
            xx, yy, zz = -sx, sz, sy
        elif ipanel == 4:
            xx, yy, zz = sx, -sz, sy
        else:
            raise ValueError(f"Panel out of range in CubedSphereXYZFromABP: {ipanel}")
        
        return xx, yy, zz
    
    def cubed_sphere_rll_from_abp(self, alpha: float, beta: float, ipanel: int) -> Tuple[float, float]:
        """Convert cubed sphere coordinates back to lat-lon"""
        xx, yy, zz = self.cubed_sphere_xyz_from_abp(alpha, beta, ipanel)
        
        # Normalize to unit sphere
        norm = math.sqrt(xx*xx + yy*yy + zz*zz)
        if norm < TINY:
            raise ValueError("Degenerate point in coordinate conversion")
        
        xx, yy, zz = xx/norm, yy/norm, zz/norm
        
        # Convert to lat-lon
        if abs(zz) > 1.0:
            zz = math.copysign(1.0, zz)  # Handle numerical precision issues
        
        lat = math.asin(zz)
        
        if xx == 0.0 and yy == 0.0:
            lon = 0.0
        else:
            lon = math.atan2(yy, xx) + ROTATE_CUBE
            # Normalize longitude to [0, 2π]
            while lon < 0.0:
                lon += 2.0 * PI
            while lon >= 2.0 * PI:
                lon -= 2.0 * PI
        
        return lon, lat
    
    def equiangular_all_areas(self) -> np.ndarray:
        """Compute areas of all cubed sphere grid cells"""
        dA = np.zeros((self.ncube, self.ncube), dtype=np.float64)
        
        # Grid points
        gp = np.array([-PIQ + (PI / (2 * self.ncube)) * k for k in range(self.ncube + 1)])
        
        # Compute angles between grid lines
        ang = np.zeros((self.ncube + 1, self.ncube + 1))
        for k1 in range(self.ncube + 1):
            for k2 in range(self.ncube + 1):
                cos_ang = -math.sin(gp[k1]) * math.sin(gp[k2])
                # Handle numerical precision
                cos_ang = max(-1.0, min(1.0, cos_ang))
                ang[k1, k2] = math.acos(cos_ang)
        
        # Compute areas using spherical excess formula
        for k1 in range(self.ncube):
            for k2 in range(self.ncube):
                a1 = ang[k1, k2]
                a2 = PI - ang[k1+1, k2]
                a3 = PI - ang[k1, k2+1]
                a4 = ang[k1+1, k2+1]
                
                # Area = r²(-2π + sum of interior angles)
                dA[k1, k2] = -2.0 * PI + a1 + a2 + a3 + a4
        
        # Consistency check
        total_area = np.sum(dA)
        expected_area = 4.0 * PI / 6.0  # One face of cube
        consistency = total_area - expected_area
        
        print(f"Area consistency check: {consistency:.6e} (should be ~0)")
        
        if abs(consistency) > 1e-10:
            print("Warning: Large area consistency error - check grid computation")
        
        return dA
    
    def bin_data_to_cubed_sphere(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bin lat-lon data to cubed-sphere grid"""
        print("Binning lat-lon data to cubed-sphere")
        
        # Grid spacing
        da = PI / (2 * self.ncube)
        
        # Convert coordinates to radians
        lon_rad = self.lon * DEG2RAD
        lat_rad = self.lat * DEG2RAD
        dlat = PI / self.jm
        
        # Initialize arrays
        weight = np.zeros((self.ncube, self.ncube, 6), dtype=np.float64)
        terr_cube = np.zeros((self.ncube, self.ncube, 6), dtype=np.float64)
        landfrac_cube = np.zeros((self.ncube, self.ncube, 6), dtype=np.float64)
        
        # Index arrays for variance computation
        idx = np.zeros((self.im, self.jm), dtype=np.int32)
        idy = np.zeros((self.im, self.jm), dtype=np.int32)
        idp = np.zeros((self.im, self.jm), dtype=np.int32)
        
        total_points = self.im * self.jm
        processed = 0
        
        for j in range(self.jm):
            for i in range(self.im):
                # Progress reporting
                if processed % (total_points // 20) == 0:
                    progress = 100.0 * processed / total_points
                    print(f"Progress: {progress:.1f}% done", end='\r')
                
                try:
                    alpha, beta, ipanel = self.cubed_sphere_abp_from_rll(lon_rad[i], lat_rad[j])
                    
                    # Find cubed sphere grid indices (1-based in original)
                    icube_1based = math.ceil((alpha + PIQ) / da)
                    jcube_1based = math.ceil((beta + PIQ) / da)
                    
                    # Convert to 0-based and validate
                    icube = icube_1based - 1
                    jcube = jcube_1based - 1
                    
                    if not (0 <= icube < self.ncube and 0 <= jcube < self.ncube):
                        print(f"\nFatal error: grid indices out of range")
                        print(f"Point ({i},{j}): lon={math.degrees(lon_rad[i]):.3f}, lat={math.degrees(lat_rad[j]):.3f}")
                        print(f"Alpha={alpha:.6f}, Beta={beta:.6f}, Panel={ipanel}")
                        print(f"Computed indices: icube={icube_1based}, jcube={jcube_1based}")
                        print(f"Valid range: [1, {self.ncube}]")
                        sys.exit(1)
                    
                    # Compute weight
                    lat_plus = lat_rad[j] + 0.5 * dlat
                    lat_minus = lat_rad[j] - 0.5 * dlat
                    wt = math.sin(lat_plus) - math.sin(lat_minus)
                    
                    if wt <= 0:
                        print(f"\nWarning: non-positive weight {wt} at ({i},{j})")
                        continue
                    
                    # Accumulate data
                    panel_idx = ipanel - 1  # Convert to 0-based
                    weight[icube, jcube, panel_idx] += wt
                    
                    if not (math.isnan(self.terr[j, i]) or self.terr[j, i] == -9999):
                        terr_cube[icube, jcube, panel_idx] += wt * float(self.terr[j, i])
                    
                    if not (math.isnan(self.landfrac[j, i]) or self.landfrac[j, i] == -99.0):
                        landfrac_cube[icube, jcube, panel_idx] += wt * float(self.landfrac[j, i])
                    
                    # Store indices for variance computation
                    idx[i, j] = icube
                    idy[i, j] = jcube
                    idp[i, j] = panel_idx
                    
                except Exception as e:
                    print(f"\nError processing point ({i},{j}): {e}")
                    sys.exit(1)
                
                processed += 1
        
        print("\nBinning completed")
        
        return weight, terr_cube, landfrac_cube, idx, idy, idp, da
    
    def interpolate_landm_coslat(self, da: float) -> np.ndarray:
        """Interpolate LANDM_COSLAT to cubed sphere grid"""
        print("Interpolating LANDM_COSLAT to cubed sphere grid")
        
        landm_coslat_cube = np.zeros((self.ncube, self.ncube, 6), dtype=np.float64)
        
        # Grid spacing in LANDM_COSLAT data
        dx_landm = DEG2RAD * (self.lon_landm[1] - self.lon_landm[0])
        
        # Check for uniform spacing
        lon_diffs = np.diff(self.lon_landm)
        if not np.allclose(lon_diffs, lon_diffs[0], rtol=1e-6):
            print("Warning: non-uniform longitude spacing in LANDM_COSLAT data")
        
        total_points = 6 * self.ncube * self.ncube
        processed = 0
        
        for k in range(6):
            for j in range(self.ncube):
                for i in range(self.ncube):
                    # Progress reporting
                    if processed % (total_points // 20) == 0:
                        progress = 100.0 * processed / total_points
                        print(f"Interpolation progress: {progress:.1f}%", end='\r')
                    
                    try:
                        # Compute center of cubed sphere cell
                        alpha = -PIQ + (i + 0.5) * da
                        beta = -PIQ + (j + 0.5) * da
                        
                        lambda_coord, theta = self.cubed_sphere_rll_from_abp(alpha, beta, k + 1)
                        
                        # Handle polar regions
                        lat_max_rad = self.lat_landm[-1] * DEG2RAD
                        lat_min_rad = self.lat_landm[0] * DEG2RAD
                        
                        if theta > lat_max_rad - TINY:
                            landm_coslat_cube[i, j, k] = 0.0
                        elif theta < lat_min_rad + TINY:
                            landm_coslat_cube[i, j, k] = 1.0
                        else:
                            # Bilinear interpolation
                            # Find longitude index
                            lon_norm = lambda_coord - self.lon_landm[0] * DEG2RAD
                            ilon = max(min(int(lon_norm / dx_landm), self.im_landm - 1), 0)
                            ip1 = (ilon + 1) % self.im_landm  # Handle wraparound
                            
                            wx = (lon_norm - ilon * dx_landm) / dx_landm
                            wx = max(0.0, min(1.0, wx))  # Clamp to [0,1]
                            
                            # Find latitude index (with search for potentially non-uniform spacing)
                            lat_norm = theta - lat_min_rad
                            dy_estimate = (lat_max_rad - lat_min_rad) / (self.jm_landm - 1)
                            ilat = max(min(int(lat_norm / dy_estimate), self.jm_landm - 2), 0)
                            
                            # Refine latitude index with search
                            search_limit = 10
                            search_count = 0
                            
                            while search_count < search_limit:
                                if ilat >= self.jm_landm - 1:
                                    ilat = self.jm_landm - 2
                                    break
                                if ilat < 0:
                                    ilat = 0
                                    break
                                
                                jp1 = ilat + 1
                                lat_ilat = self.lat_landm[ilat] * DEG2RAD
                                lat_jp1 = self.lat_landm[jp1] * DEG2RAD
                                
                                if lat_ilat <= theta <= lat_jp1:
                                    break
                                elif theta > lat_jp1:
                                    ilat += 1
                                else:
                                    ilat -= 1
                                
                                search_count += 1
                            
                            if search_count >= search_limit:
                                print(f"\nWarning: latitude search failed at ({i},{j},{k})")
                                landm_coslat_cube[i, j, k] = 0.5
                                processed += 1
                                continue
                            
                            jp1 = ilat + 1
                            lat_ilat = self.lat_landm[ilat] * DEG2RAD
                            lat_jp1 = self.lat_landm[jp1] * DEG2RAD
                            
                            if lat_jp1 - lat_ilat == 0:
                                wy = 0.0
                            else:
                                wy = (theta - lat_ilat) / (lat_jp1 - lat_ilat)
                                wy = max(0.0, min(1.0, wy))  # Clamp to [0,1]
                            
                            # Bounds checking
                            if not (0 <= ilon < self.im_landm and 0 <= ilat < self.jm_landm-1):
                                print(f"\nError: interpolation indices out of bounds at ({i},{j},{k})")
                                print(f"ilon={ilon}, ilat={ilat}, im_landm={self.im_landm}, jm_landm={self.jm_landm}")
                                landm_coslat_cube[i, j, k] = 0.5
                                processed += 1
                                continue
                            
                            # Perform bilinear interpolation
                            try:
                                val00 = self.landm_coslat[ilat, ilon]
                                val01 = self.landm_coslat[ilat, ip1]
                                val10 = self.landm_coslat[jp1, ilon]
                                val11 = self.landm_coslat[jp1, ip1]
                                
                                # Check for fill values
                                if (abs(val00) > 999999 or abs(val01) > 999999 or 
                                    abs(val10) > 999999 or abs(val11) > 999999):
                                    landm_coslat_cube[i, j, k] = 0.5
                                else:
                                    landm_coslat_cube[i, j, k] = (
                                        (1.0 - wx) * (1.0 - wy) * val00 +
                                        wx * (1.0 - wy) * val01 +
                                        (1.0 - wx) * wy * val10 +
                                        wx * wy * val11
                                    )
                            except IndexError as e:
                                print(f"\nIndexError in interpolation at ({i},{j},{k}): {e}")
                                landm_coslat_cube[i, j, k] = 0.5
                    
                    except Exception as e:
                        print(f"\nError in LANDM_COSLAT interpolation at ({i},{j},{k}): {e}")
                        landm_coslat_cube[i, j, k] = 0.5
                    
                    processed += 1
        
        print("\nLANDM_COSLAT interpolation completed")
        return landm_coslat_cube
    
    def normalize_and_validate(self, weight: np.ndarray, terr_cube: np.ndarray, 
                              landfrac_cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize cubed sphere data by weights and validate"""
        print("Normalizing cubed sphere data")
        
        empty_cells = 0
        total_cells = 6 * self.ncube * self.ncube
        
        for k in range(6):
            for j in range(self.ncube):
                for i in range(self.ncube):
                    if abs(weight[i, j, k]) < 1.0e-9:
                        empty_cells += 1
                        print(f"FATAL ERROR: no lat-lon grid point in cubed sphere cell ({i},{j},{k})")
                        print(f"This indicates insufficient input resolution or coordinate transformation error")
                        print(f"Input resolution: {self.im} x {self.jm}")
                        print(f"Target cubed sphere resolution: {self.ncube}")
                        print(f"Consider reducing ncube or increasing input data resolution")
                        sys.exit(1)
                    else:
                        terr_cube[i, j, k] /= weight[i, j, k]
                        landfrac_cube[i, j, k] /= weight[i, j, k]
        
        print(f"All {total_cells} cubed sphere cells successfully populated")
        
        # Validate results
        valid_terr = ~np.isnan(terr_cube) & ~np.isinf(terr_cube)
        valid_landfrac = ~np.isnan(landfrac_cube) & ~np.isinf(landfrac_cube)
        
        if not np.all(valid_terr):
            print(f"Warning: {np.sum(~valid_terr)} invalid terrain values")
        if not np.all(valid_landfrac):
            print(f"Warning: {np.sum(~valid_landfrac)} invalid land fraction values")
        
        print(f"min/max value of terr_cube: {np.min(terr_cube):.1f}, {np.max(terr_cube):.1f}")
        print(f"min/max value of landfrac_cube: {np.min(landfrac_cube):.6f}, {np.max(landfrac_cube):.6f}")
        
        return terr_cube, landfrac_cube
    
    def compute_cubed_sphere_volume(self, terr_cube: np.ndarray) -> float:
        """Compute volume for cubed-sphere data"""
        print("Computing volume for cubed-sphere binned data")
        
        darea_cube = self.equiangular_all_areas()
        vol_cube = 0.0
        
        for ipanel in range(6):
            for j in range(self.ncube):
                for i in range(self.ncube):
                    if not (math.isnan(terr_cube[i, j, ipanel]) or math.isinf(terr_cube[i, j, ipanel])):
                        vol_cube += terr_cube[i, j, ipanel] * darea_cube[i, j]
        
        vol_cube = vol_cube / (4.0 * PI)
        return vol_cube
    
    def compute_variance(self, weight: np.ndarray, terr_cube: np.ndarray, 
                        idx: np.ndarray, idy: np.ndarray, idp: np.ndarray) -> np.ndarray:
        """Compute variance of elevation from high-res lat-lon to cubed-sphere"""
        print("Computing variance")
        
        var30_cube = np.zeros((self.ncube, self.ncube, 6), dtype=np.float64)
        
        # Convert coordinates to radians for consistency
        lat_rad = self.lat * DEG2RAD
        dlat = PI / self.jm
        
        total_points = self.im * self.jm
        processed = 0
        
        for j in range(self.jm):
            for i in range(self.im):
                # Progress reporting
                if processed % (total_points // 20) == 0:
                    progress = 100.0 * processed / total_points
                    print(f"Variance computation progress: {progress:.1f}%", end='\r')
                
                icube = idx[i, j]
                jcube = idy[i, j]
                ipanel = idp[i, j]
                
                # Check bounds
                if not (0 <= icube < self.ncube and 0 <= jcube < self.ncube and 0 <= ipanel < 6):
                    print(f"\nError: invalid indices for variance computation at ({i},{j})")
                    print(f"icube={icube}, jcube={jcube}, ipanel={ipanel}")
                    continue
                
                # Skip invalid terrain values
                if math.isnan(self.terr[j, i]) or self.terr[j, i] == -9999:
                    processed += 1
                    continue
                
                wt = math.sin(lat_rad[j] + 0.5 * dlat) - math.sin(lat_rad[j] - 0.5 * dlat)
                
                if weight[icube, jcube, ipanel] > 0:
                    diff = terr_cube[icube, jcube, ipanel] - self.terr[j, i]
                    var30_cube[icube, jcube, ipanel] += (wt * diff**2) / weight[icube, jcube, ipanel]
                
                processed += 1
        
        print(f"\nmin/max value of var30_cube: {np.min(np.sqrt(var30_cube)):.2f}, {np.max(np.sqrt(var30_cube)):.2f}")
        
        return var30_cube
    
    def write_output_file(self, terr_cube: np.ndarray, landfrac_cube: np.ndarray, 
                         landm_coslat_cube: np.ndarray, var30_cube: np.ndarray):
        """Write cubed sphere data to NetCDF file"""
        print(f"Creating NetCDF file for output: {self.output_file}")
        
        # Flatten arrays for output (Fortran column-major order)
        grid_dims = 6 * self.ncube * self.ncube
        
        # Compute grid center coordinates
        grid_center_lat = np.zeros(grid_dims, dtype=np.float64)
        grid_center_lon = np.zeros(grid_dims, dtype=np.float64)
        
        da = PI / (2 * self.ncube)
        atm_add = 0
        
        for k in range(6):
            for j in range(self.ncube):
                ygno_ce = -PIQ + da * (j + 0.5)
                for i in range(self.ncube):
                    xgno_ce = -PIQ + da * (i + 0.5)
                    try:
                        lon, lat = self.cubed_sphere_rll_from_abp(xgno_ce, ygno_ce, k + 1)
                        grid_center_lon[atm_add] = lon * RAD2DEG
                        grid_center_lat[atm_add] = lat * RAD2DEG
                    except Exception as e:
                        print(f"Error computing grid coordinates for cell ({i},{j},{k}): {e}")
                        grid_center_lon[atm_add] = 0.0
                        grid_center_lat[atm_add] = 0.0
                    
                    atm_add += 1
        
        # Flatten data arrays (using Fortran-style column-major order)
        terr_flat = np.zeros(grid_dims, dtype=np.float64)
        landfrac_flat = np.zeros(grid_dims, dtype=np.float64)
        landm_coslat_flat = np.zeros(grid_dims, dtype=np.float64)
        var30_flat = np.zeros(grid_dims, dtype=np.float64)
        
        atm_add = 0
        for k in range(6):
            for j in range(self.ncube):
                for i in range(self.ncube):
                    terr_flat[atm_add] = terr_cube[i, j, k]
                    landfrac_flat[atm_add] = landfrac_cube[i, j, k]
                    landm_coslat_flat[atm_add] = landm_coslat_cube[i, j, k]
                    var30_flat[atm_add] = var30_cube[i, j, k]
                    atm_add += 1
        
        try:
            with nc.Dataset(self.output_file, 'w', format='NETCDF4') as ncfile:
                # Global attributes
                ncfile.title = 'equi-angular gnomonic cubed sphere grid'
                ncfile.raw_topo = self.raw_latlon_data_file
                ncfile.source_code = 'https://github.com/NCAR/Topo.git'
                ncfile.note = 'LANDM_COSLAT is only used in CAM4'
                ncfile.history = f'Written on date: {datetime.now().strftime("%Y%m%d")}'
                ncfile.author = 'Peter Hjort Lauritzen (NCAR)'
                ncfile.python_conversion = 'Converted from Fortran to Python'
                ncfile.ncube_resolution = self.ncube
                
                # Dimensions
                ncfile.createDimension('grid_size', grid_dims)
                ncfile.createDimension('grid_rank', 1)
                
                # Variables
                grid_dims_var = ncfile.createVariable('grid_dims', 'i4', ('grid_rank',))
                lat_var = ncfile.createVariable('lat', 'f8', ('grid_size',))
                lon_var = ncfile.createVariable('lon', 'f8', ('grid_size',))
                terr_var = ncfile.createVariable('terr', 'f8', ('grid_size',))
                landfrac_var = ncfile.createVariable('LANDFRAC', 'f8', ('grid_size',))
                landm_coslat_var = ncfile.createVariable('LANDM_COSLAT', 'f8', ('grid_size',))
                var30_var = ncfile.createVariable('var30', 'f8', ('grid_size',))
                
                # Variable attributes
                lat_var.units = 'degrees_north'
                lat_var.long_name = 'latitude of grid cell centers'
                
                lon_var.units = 'degrees_east'
                lon_var.long_name = 'longitude of grid cell centers'
                
                terr_var.units = 'm'
                terr_var.long_name = 'surface elevation'
                terr_var.standard_name = 'surface_altitude'
                
                landfrac_var.long_name = 'land ocean transition mask: ocean (0), continent (1), transition (0-1)'
                landfrac_var.units = '1'
                
                landm_coslat_var.long_name = 'smoothed land ocean transition mask'
                landm_coslat_var.units = '1'
                landm_coslat_var.note = 'only used in CAM4'
                
                var30_var.units = 'm^2'
                var30_var.long_name = 'variance of elevation from high res lat-lon to ~3km cubed-sphere'
                var30_var.description = 'subgrid variance of surface elevation'
                
                # Write data
                grid_dims_var[:] = grid_dims
                lat_var[:] = grid_center_lat
                lon_var[:] = grid_center_lon
                terr_var[:] = terr_flat
                landfrac_var[:] = landfrac_flat
                landm_coslat_var[:] = landm_coslat_flat
                var30_var[:] = var30_flat
                
                # Add some diagnostic information
                ncfile.input_dimensions = f'{self.im} x {self.jm}'
                ncfile.output_dimensions = f'{self.ncube} x {self.ncube} x 6'
                ncfile.total_input_points = self.im * self.jm
                ncfile.total_output_points = grid_dims
                
        except Exception as e:
            print(f"Error writing output file: {e}")
            sys.exit(1)
        
        print(f"Successfully wrote output file: {self.output_file}")
    
    def run(self):
        """Main processing pipeline"""
        try:
            # Read input data
            self.read_terrain_data()
            self.read_landm_coslat_data()
            
            # Compute volume for validation
            vol_raw = self.compute_raw_volume()
            
            # Bin data to cubed sphere
            weight, terr_cube, landfrac_cube, idx, idy, idp, da = self.bin_data_to_cubed_sphere()
            
            # Normalize by weights
            terr_cube, landfrac_cube = self.normalize_and_validate(weight, terr_cube, landfrac_cube)
            
            # Interpolate LANDM_COSLAT
            landm_coslat_cube = self.interpolate_landm_coslat(da)
            
            print(f"min/max value of landm_coslat_cube: {np.min(landm_coslat_cube):.6f}, {np.max(landm_coslat_cube):.6f}")
            
            # Compute volume for cubed sphere data
            vol_cube = self.compute_cubed_sphere_volume(terr_cube)
            print(f"Mean height (globally) of topography about sea-level:")
            print(f"  Raw data: {vol_raw:.2f} m")
            print(f"  Cubed sphere data: {vol_cube:.2f} m")
            print(f"  Relative difference: {(vol_cube-vol_raw)/vol_raw*100:.3f}%")
            
            # Compute variance
            var30_cube = self.compute_variance(weight, terr_cube, idx, idy, idp)
            
            # Write output
            self.write_output_file(terr_cube, landfrac_cube, landm_coslat_cube, var30_cube)
            
            print("Processing completed successfully!")
            
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"Fatal error during processing: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert lat-lon terrain data to cubed-sphere grid',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bin_to_cube_2ndgen.py
  python bin_to_cube_2ndgen.py --config my_config.ini
  python bin_to_cube_2ndgen.py --ncube 1500 --input terrain.nc --output cube_terrain.nc

Config file format (INI):
  [binparams]
  raw_latlon_data_file = input_terrain.nc
  output_file = output_cubed_sphere.nc
  ncube = 3000
  landm_coslat_file = landm_coslat.nc
        """
    )
    
    parser.add_argument('--config', '-c', default='bin_to_cube.ini',
                       help='Configuration file (default: bin_to_cube.ini)')
    parser.add_argument('--ncube', '-n', type=int,
                       help='Cubed sphere resolution (overrides config file)')
    parser.add_argument('--input', '-i', 
                       help='Input terrain NetCDF file (overrides config file)')
    parser.add_argument('--output', '-o',
                       help='Output NetCDF file (overrides config file)')
    parser.add_argument('--landm-coslat', '-l',
                       help='LANDM_COSLAT NetCDF file (overrides config file)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        # Create processor
        processor = TerrainProcessor(args.config)
        
        # Override config with command line arguments
        if args.ncube:
            processor.ncube = args.ncube
            processor.config['ncube'] = args.ncube
        if args.input:
            processor.raw_latlon_data_file = args.input
            processor.config['raw_latlon_data_file'] = args.input
        if args.output:
            processor.output_file = args.output
            processor.config['output_file'] = args.output
        if args.landm_coslat:
            processor.landm_coslat_file = args.landm_coslat
            processor.config['landm_coslat_file'] = args.landm_coslat
        
        print("=" * 60)
        print("Terrain to Cubed Sphere Converter")
        print("=" * 60)
        print(f"Input file: {processor.raw_latlon_data_file}")
        print(f"LANDM_COSLAT file: {processor.landm_coslat_file}")
        print(f"Output file: {processor.output_file}")
        print(f"Cubed sphere resolution: {processor.ncube}")
        print("=" * 60)
        
        # Run processing
        processor.run()
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
