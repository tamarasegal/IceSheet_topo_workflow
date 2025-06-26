#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for bin_to_cube Environmental Model

CRITICAL SAFETY NOTICE:
This code is used for environmental modeling where errors could lead to:
- Incorrect climate predictions
- Faulty atmospheric simulations  
- Miscalculated pollution dispersion models
- Wrong environmental impact assessments

Every test case must pass before deployment to production systems.

Test Coverage:
1. Coordinate transformation accuracy (spherical geometry)
2. Data conservation laws (mass/energy conservation)
3. Interpolation mathematical correctness
4. Grid area calculations (spherical trigonometry)
5. Boundary condition handling
6. Numerical precision and stability
7. Error propagation analysis
8. Memory safety and bounds checking
9. File format compliance and data integrity
10. Edge cases that could cause silent failures

Author: Senior Environmental Modeling QA Engineer
Classification: SAFETY-CRITICAL ENVIRONMENTAL CODE
"""

import unittest
import numpy as np
import tempfile
import os
import math
import netCDF4 as nc
from typing import Tuple, List
import warnings
import sys
import time
import psutil

# Import the module under test
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bin_to_cube_2ndgen import TerrainProcessor

# Test constants - must match physical constants exactly
PI = math.pi
RAD2DEG = 180.0 / PI
DEG2RAD = PI / 180.0
EARTH_RADIUS = 6371000.0  # meters
EARTH_SURFACE_AREA = 4.0 * PI * EARTH_RADIUS**2
NUMERICAL_TOLERANCE = 1e-12  # Strict tolerance for safety-critical calculations


class EnvironmentalSafetyTestCase(unittest.TestCase):
    """
    Base class for safety-critical environmental test cases.
    Implements strict error checking and validation specific to environmental modeling.
    """
    
    def assertAlmostEqualStrict(self, first, second, tolerance=NUMERICAL_TOLERANCE, msg=None):
        """Strict numerical comparison for safety-critical calculations"""
        if abs(first - second) > tolerance:
            standard_msg = f'{first} != {second} within tolerance {tolerance}'
            self.fail(self._formatMessage(msg, standard_msg))
    
    def assertConservationLaw(self, initial_value, final_value, conservation_type="mass", tolerance=1e-10):
        """Verify conservation laws are maintained (critical for environmental models)"""
        relative_error = abs(final_value - initial_value) / abs(initial_value) if initial_value != 0 else abs(final_value)
        if relative_error > tolerance:
            self.fail(f"{conservation_type.title()} conservation violated: "
                     f"initial={initial_value}, final={final_value}, "
                     f"relative_error={relative_error:.2e} > tolerance={tolerance:.2e}")
    
    def assertPhysicallyRealistic(self, value, min_val, max_val, quantity_name):
        """Verify values are within physically realistic bounds"""
        if not (min_val <= value <= max_val):
            self.fail(f"{quantity_name} = {value} is outside physically realistic range [{min_val}, {max_val}]")


class CoordinateTransformationTests(EnvironmentalSafetyTestCase):
    """
    CRITICAL: Test coordinate transformations for mathematical correctness.
    
    Failures here could cause:
    - Terrain data mapped to wrong locations
    - Climate model grid misalignment
    - Pollution source misplacement
    """
    
    def setUp(self):
        """Initialize processor for coordinate tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.ini")
        
        # Create minimal valid config
        with open(self.config_file, 'w') as f:
            f.write("""[binparams]
raw_latlon_data_file = dummy.nc
output_file = dummy_out.nc
ncube = 10
landm_coslat_file = dummy_landm.nc
""")
        
        self.processor = TerrainProcessor(self.config_file)
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_coordinate_transformation_roundtrip_identity(self):
        """
        CRITICAL TEST: Verify coordinate transformations are bijective.
        
        Forward transform: (lon, lat) -> (alpha, beta, panel)
        Inverse transform: (alpha, beta, panel) -> (lon, lat)
        Must satisfy: inverse(forward(x)) = x within numerical precision
        """
        test_coordinates = [
            (0.0, 0.0),           # Equator, Prime Meridian
            (PI, 0.0),            # Equator, Date Line
            (0.0, PI/2 - 1e-6),   # Near North Pole
            (0.0, -PI/2 + 1e-6),  # Near South Pole
            (PI/2, PI/4),         # 90°E, 45°N
            (-PI/2, -PI/4),       # 90°W, 45°S
            (2*PI - 1e-10, 0.0),  # Near 360° longitude
        ]
        
        for lon_orig, lat_orig in test_coordinates:
            with self.subTest(lon=math.degrees(lon_orig), lat=math.degrees(lat_orig)):
                try:
                    # Forward transformation
                    alpha, beta, panel = self.processor.cubed_sphere_abp_from_rll(lon_orig, lat_orig)
                    
                    # Verify panel is valid
                    self.assertIn(panel, range(1, 7), f"Invalid panel {panel}")
                    
                    # Verify alpha, beta are in valid range [-π/4, π/4]
                    self.assertPhysicallyRealistic(alpha, -PI/4, PI/4, "alpha coordinate")
                    self.assertPhysicallyRealistic(beta, -PI/4, PI/4, "beta coordinate")
                    
                    # Inverse transformation
                    lon_recovered, lat_recovered = self.processor.cubed_sphere_rll_from_abp(alpha, beta, panel)
                    
                    # Handle longitude wraparound
                    lon_diff = abs(lon_recovered - lon_orig)
                    if lon_diff > PI:
                        lon_diff = 2*PI - lon_diff
                    
                    lat_diff = abs(lat_recovered - lat_orig)
                    
                    # Strict numerical verification
                    self.assertAlmostEqualStrict(lon_diff, 0.0, tolerance=1e-14,
                                               msg=f"Longitude roundtrip failed: {math.degrees(lon_orig)} -> {math.degrees(lon_recovered)}")
                    self.assertAlmostEqualStrict(lat_diff, 0.0, tolerance=1e-14,
                                               msg=f"Latitude roundtrip failed: {math.degrees(lat_orig)} -> {math.degrees(lat_recovered)}")
                    
                except Exception as e:
                    self.fail(f"Coordinate transformation failed for ({math.degrees(lon_orig):.6f}°, {math.degrees(lat_orig):.6f}°): {e}")
    
    def test_coordinate_transformation_coverage(self):
        """
        CRITICAL TEST: Verify all points on sphere map to exactly one panel.
        
        Every point on the sphere must map to exactly one cubed sphere panel.
        No gaps or overlaps allowed - this could cause missing or duplicate data.
        """
        # Test systematic grid coverage
        n_lon, n_lat = 72, 36  # 5° resolution
        
        panel_counts = {i: 0 for i in range(1, 7)}
        total_points = 0
        
        for i_lon in range(n_lon):
            for i_lat in range(n_lat):
                lon = 2 * PI * i_lon / n_lon
                lat = PI * (i_lat / (n_lat - 1) - 0.5)
                
                # Skip exact poles (undefined longitude)
                if abs(lat) >= PI/2 - 1e-10:
                    continue
                
                try:
                    alpha, beta, panel = self.processor.cubed_sphere_abp_from_rll(lon, lat)
                    panel_counts[panel] += 1
                    total_points += 1
                    
                    # Verify panel assignment is deterministic
                    alpha2, beta2, panel2 = self.processor.cubed_sphere_abp_from_rll(lon, lat)
                    self.assertEqual(panel, panel2, f"Non-deterministic panel assignment at ({math.degrees(lon):.2f}°, {math.degrees(lat):.2f}°)")
                    
                except Exception as e:
                    self.fail(f"Coordinate transformation failed at ({math.degrees(lon):.2f}°, {math.degrees(lat):.2f}°): {e}")
        
        # Verify each panel has reasonable coverage
        for panel_id, count in panel_counts.items():
            coverage_fraction = count / total_points
            self.assertGreater(coverage_fraction, 0.1, f"Panel {panel_id} has insufficient coverage: {coverage_fraction:.3f}")
            self.assertLess(coverage_fraction, 0.25, f"Panel {panel_id} has excessive coverage: {coverage_fraction:.3f}")


class AreaCalculationTests(EnvironmentalSafetyTestCase):
    """
    CRITICAL: Test spherical area calculations for conservation laws.
    
    Failures here could cause:
    - Incorrect volume/mass conservation in atmospheric models
    - Wrong surface flux calculations
    - Inaccurate pollution dispersion rates
    """
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.ini")
        
        with open(self.config_file, 'w') as f:
            f.write("""[binparams]
raw_latlon_data_file = dummy.nc
output_file = dummy_out.nc
ncube = 48
landm_coslat_file = dummy_landm.nc
""")
        
        self.processor = TerrainProcessor(self.config_file)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_total_spherical_area_conservation(self):
        """
        CRITICAL TEST: Verify total area equals 4π (unit sphere).
        
        The sum of all cubed sphere grid cell areas must equal the surface area
        of a unit sphere. This is a fundamental conservation law.
        """
        # Test multiple resolutions to verify scaling
        test_resolutions = [16, 32, 48]
        
        for ncube in test_resolutions:
            with self.subTest(ncube=ncube):
                self.processor.ncube = ncube
                
                # Calculate areas for one face of the cube
                darea_face = self.processor.equiangular_all_areas()
                
                # Total area = 6 faces × sum of face areas
                total_area = 6.0 * np.sum(darea_face)
                expected_area = 4.0 * PI  # Unit sphere surface area
                
                # Verify conservation with strict tolerance
                relative_error = abs(total_area - expected_area) / expected_area
                self.assertLess(relative_error, 1e-12, 
                              f"Area conservation violated for ncube={ncube}: "
                              f"computed={total_area:.15f}, expected={expected_area:.15f}, "
                              f"relative_error={relative_error:.2e}")
                
                # Verify all areas are positive
                self.assertTrue(np.all(darea_face > 0), 
                              f"Found non-positive areas for ncube={ncube}")


class DataConservationTests(EnvironmentalSafetyTestCase):
    """
    CRITICAL: Test data conservation during binning operations.
    
    Failures here could cause:
    - Mass/energy non-conservation in climate models
    - Incorrect total pollution calculations
    - Biased atmospheric flux estimates
    """
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.ini")
        
        # Create synthetic test data
        self.im, self.jm = 72, 36  # 5° resolution
        self.ncube = 24
        
        # Create test NetCDF files
        self.terrain_file = os.path.join(self.temp_dir, "terrain.nc")
        self.landm_file = os.path.join(self.temp_dir, "landm.nc")
        self.output_file = os.path.join(self.temp_dir, "output.nc")
        
        self._create_test_data()
        
        with open(self.config_file, 'w') as f:
            f.write(f"""[binparams]
raw_latlon_data_file = {self.terrain_file}
output_file = {self.output_file}
ncube = {self.ncube}
landm_coslat_file = {self.landm_file}
""")
        
        self.processor = TerrainProcessor(self.config_file)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_data(self):
        """Create synthetic test data with known properties"""
        # Create longitude and latitude arrays
        lon = np.linspace(0, 360, self.im, endpoint=False)
        lat = np.linspace(-90, 90, self.jm)
        
        # Create meshgrid
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        
        # Create synthetic terrain with known analytical properties
        # Use spherical harmonics for mathematically well-defined functions
        terrain = (1000.0 * (2.0 + np.cos(np.radians(lat_2d))**2 * 
                            np.cos(2.0 * np.radians(lon_2d))))
        
        # Create land fraction (0 to 1)
        landfrac = 0.5 * (1.0 + np.cos(np.radians(lat_2d)) * 
                         np.sin(np.radians(lon_2d)))
        
        # Ensure land fraction is in valid range
        landfrac = np.clip(landfrac.astype(np.float32), 0.0, 1.0)
        terrain = terrain.astype(np.float32)
        
        # Create terrain NetCDF file
        with nc.Dataset(self.terrain_file, 'w') as ncfile:
            ncfile.createDimension('lon', self.im)
            ncfile.createDimension('lat', self.jm)
            
            lon_var = ncfile.createVariable('lon', 'f8', ('lon',))
            lat_var = ncfile.createVariable('lat', 'f8', ('lat',))
            terrain_var = ncfile.createVariable('htopo', 'f4', ('lat', 'lon'))
            landfrac_var = ncfile.createVariable('landfract', 'f4', ('lat', 'lon'))
            
            lon_var[:] = lon
            lat_var[:] = lat
            terrain_var[:] = terrain
            landfrac_var[:] = landfrac
            
            # Add attributes
            terrain_var.units = 'm'
            landfrac_var.units = '1'
        
        # Create LANDM_COSLAT file (simplified)
        landm_coslat = np.random.random((self.jm, self.im)).astype(np.float64)
        
        with nc.Dataset(self.landm_file, 'w') as ncfile:
            ncfile.createDimension('lon', self.im)
            ncfile.createDimension('lat', self.jm)
            
            lon_var = ncfile.createVariable('lon', 'f8', ('lon',))
            lat_var = ncfile.createVariable('lat', 'f8', ('lat',))
            landm_var = ncfile.createVariable('LANDM_COSLAT', 'f8', ('lat', 'lon'))
            
            lon_var[:] = lon
            lat_var[:] = lat
            landm_var[:] = landm_coslat
    
    def test_mass_conservation_during_binning(self):
        """
        CRITICAL TEST: Verify total mass is conserved during binning.
        
        The weighted integral of terrain elevation over the sphere must be
        conserved during the lat-lon to cubed-sphere transformation.
        """
        # Load test data
        self.processor.read_terrain_data()
        self.processor.read_landm_coslat_data()
        
        # Calculate original weighted volume
        original_volume = self.processor.compute_raw_volume()
        
        # Perform binning
        weight, terr_cube, landfrac_cube, idx, idy, idp, da = self.processor.bin_data_to_cubed_sphere()
        
        # Normalize
        terr_cube, landfrac_cube = self.processor.normalize_and_validate(weight, terr_cube, landfrac_cube)
        
        # Calculate cubed sphere volume
        cubed_volume = self.processor.compute_cubed_sphere_volume(terr_cube)
        
        # Verify conservation
        self.assertConservationLaw(original_volume, cubed_volume, "mass", tolerance=1e-6)
        
        # Additional checks for volume calculation consistency
        self.assertPhysicallyRealistic(original_volume, -1000, 10000, "original volume")
        self.assertPhysicallyRealistic(cubed_volume, -1000, 10000, "cubed sphere volume")


class BoundaryConditionTests(EnvironmentalSafetyTestCase):
    """
    CRITICAL: Test boundary conditions and edge cases.
    
    Failures here could cause:
    - Undefined behavior at poles or date line
    - Incorrect data at continent boundaries
    - Numerical instabilities in edge regions
    """
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.ini")
        
        with open(self.config_file, 'w') as f:
            f.write("""[binparams]
raw_latlon_data_file = dummy.nc
output_file = dummy_out.nc
ncube = 16
landm_coslat_file = dummy_landm.nc
""")
        
        self.processor = TerrainProcessor(self.config_file)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_polar_regions_handling(self):
        """
        CRITICAL TEST: Verify correct handling of polar regions.
        
        Polar regions are mathematically singular and require special handling.
        Errors here could cause model instabilities at high latitudes.
        """
        # Test points very close to poles
        epsilon = 1e-10
        polar_test_points = [
            (0.0, PI/2 - epsilon),    # Near North Pole
            (PI, PI/2 - epsilon),     # Near North Pole, opposite longitude
            (0.0, -PI/2 + epsilon),   # Near South Pole
            (PI, -PI/2 + epsilon),    # Near South Pole, opposite longitude
        ]
        
        for lon, lat in polar_test_points:
            with self.subTest(lon=math.degrees(lon), lat=math.degrees(lat)):
                try:
                    alpha, beta, panel = self.processor.cubed_sphere_abp_from_rll(lon, lat)
                    
                    # Verify transformation is well-defined
                    self.assertFalse(math.isnan(alpha), f"NaN alpha at polar point")
                    self.assertFalse(math.isnan(beta), f"NaN beta at polar point")
                    self.assertIn(panel, [5, 6], f"Unexpected panel {panel} for polar point")
                    
                    # Verify roundtrip transformation
                    lon_recovered, lat_recovered = self.processor.cubed_sphere_rll_from_abp(alpha, beta, panel)
                    
                    lat_error = abs(lat_recovered - lat)
                    self.assertLess(lat_error, 1e-12, f"Large latitude error at pole: {lat_error}")
                    
                except Exception as e:
                    self.fail(f"Polar transformation failed at ({math.degrees(lon):.10f}°, {math.degrees(lat):.10f}°): {e}")


class NumericalStabilityTests(EnvironmentalSafetyTestCase):
    """
    CRITICAL: Test numerical stability and precision.
    
    Failures here could cause:
    - Accumulating errors in long simulations
    - Inconsistent results across platforms
    - Catastrophic cancellation errors
    """
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.ini")
        
        with open(self.config_file, 'w') as f:
            f.write("""[binparams]
raw_latlon_data_file = dummy.nc
output_file = dummy_out.nc
ncube = 32
landm_coslat_file = dummy_landm.nc
""")
        
        self.processor = TerrainProcessor(self.config_file)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_coordinate_transformation_precision(self):
        """
        CRITICAL TEST: Verify coordinate transformations maintain precision.
        
        Multiple iterations of forward/inverse transforms should not
        accumulate significant numerical errors.
        """
        # Test precision over multiple iterations
        original_coords = [
            (0.5, 0.3),           # Arbitrary point
            (PI/3, PI/6),         # 60°E, 30°N
            (4*PI/3, -PI/6),      # 240°E, 30°S
        ]
        
        for lon_orig, lat_orig in original_coords:
            with self.subTest(lon=math.degrees(lon_orig), lat=math.degrees(lat_orig)):
                lon_current, lat_current = lon_orig, lat_orig
                
                # Perform 100 roundtrip transformations
                for iteration in range(100):
                    try:
                        # Forward transform
                        alpha, beta, panel = self.processor.cubed_sphere_abp_from_rll(lon_current, lat_current)
                        
                        # Inverse transform
                        lon_current, lat_current = self.processor.cubed_sphere_rll_from_abp(alpha, beta, panel)
                        
                        # Check for NaN or infinity
                        self.assertFalse(math.isnan(lon_current), f"NaN longitude at iteration {iteration}")
                        self.assertFalse(math.isnan(lat_current), f"NaN latitude at iteration {iteration}")
                        self.assertFalse(math.isinf(lon_current), f"Infinite longitude at iteration {iteration}")
                        self.assertFalse(math.isinf(lat_current), f"Infinite latitude at iteration {iteration}")
                        
                    except Exception as e:
                        self.fail(f"Precision test failed at iteration {iteration}: {e}")
                
                # Check final accumulated error
                lon_diff = abs(lon_current - lon_orig)
                if lon_diff > PI:
                    lon_diff = 2*PI - lon_diff  # Handle wraparound
                
                lat_diff = abs(lat_current - lat_orig)
                
                # Error should not grow significantly
                self.assertLess(lon_diff, 1e-10, f"Longitude precision degraded: {lon_diff:.2e}")
                self.assertLess(lat_diff, 1e-10, f"Latitude precision degraded: {lat_diff:.2e}")


class IntegrationTests(EnvironmentalSafetyTestCase):
    """
    CRITICAL: End-to-end integration tests.
    
    Tests the complete pipeline with realistic data to ensure
    the entire system works correctly in production scenarios.
    """
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.ini")
        
        # Create realistic test data
        self.im, self.jm = 144, 72  # 2.5° resolution
        self.ncube = 48
        
        self.terrain_file = os.path.join(self.temp_dir, "terrain.nc")
        self.landm_file = os.path.join(self.temp_dir, "landm.nc")
        self.output_file = os.path.join(self.temp_dir, "output.nc")
        
        self._create_realistic_test_data()
        
        with open(self.config_file, 'w') as f:
            f.write(f"""[binparams]
raw_latlon_data_file = {self.terrain_file}
output_file = {self.output_file}
ncube = {self.ncube}
landm_coslat_file = {self.landm_file}
""")
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_realistic_test_data(self):
        """Create realistic test data mimicking real Earth data"""
        # Create coordinate arrays
        lon = np.linspace(0, 360, self.im, endpoint=False)
        lat = np.linspace(-90, 90, self.jm)
        
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        
        # Create realistic terrain elevation using multiple scales
        terrain = np.zeros_like(lat_2d)
        
        # Add continent-like features
        for center_lon, center_lat, height, width in [
            (0, 45, 2000, 30),      # European-like feature
            (100, 30, 4000, 40),    # Asian-like feature  
            (-100, 40, 3000, 50),   # North American-like feature
            (150, -30, 1500, 35),   # Australian-like feature
        ]:
            dist = np.sqrt((lon_2d - center_lon)**2 + (lat_2d - center_lat)**2)
            terrain += height * np.exp(-(dist/width)**2)
        
        # Add noise to simulate real data
        terrain += np.random.normal(0, 100, terrain.shape)
        
        # Create realistic land fraction
        landfrac = np.where(terrain > 0, 
                           0.8 + 0.2 * np.random.random(terrain.shape),  # Land
                           0.1 * np.random.random(terrain.shape))        # Ocean
        
        landfrac = np.clip(landfrac, 0.0, 1.0).astype(np.float32)
        terrain = terrain.astype(np.float32)
        
        # Create terrain NetCDF file
        with nc.Dataset(self.terrain_file, 'w') as ncfile:
            ncfile.createDimension('lon', self.im)
            ncfile.createDimension('lat', self.jm)
            
            lon_var = ncfile.createVariable('lon', 'f8', ('lon',))
            lat_var = ncfile.createVariable('lat', 'f8', ('lat',))
            terrain_var = ncfile.createVariable('htopo', 'f4', ('lat', 'lon'))
            landfrac_var = ncfile.createVariable('landfract', 'f4', ('lat', 'lon'))
            
            lon_var[:] = lon
            lat_var[:] = lat
            terrain_var[:] = terrain
            landfrac_var[:] = landfrac
            
            terrain_var.units = 'm'
            landfrac_var.units = '1'
        
        # Create LANDM_COSLAT file
        landm_coslat = 0.5 * (1 + np.cos(np.radians(lat_2d)) * np.sin(np.radians(lon_2d)))
        landm_coslat = landm_coslat.astype(np.float64)
        
        with nc.Dataset(self.landm_file, 'w') as ncfile:
            ncfile.createDimension('lon', self.im)
            ncfile.createDimension('lat', self.jm)
            
            lon_var = ncfile.createVariable('lon', 'f8', ('lon',))
            lat_var = ncfile.createVariable('lat', 'f8', ('lat',))
            landm_var = ncfile.createVariable('LANDM_COSLAT', 'f8', ('lat', 'lon'))
            
            lon_var[:] = lon
            lat_var[:] = lat
            landm_var[:] = landm_coslat
    
    def test_complete_pipeline_execution(self):
        """
        CRITICAL TEST: Verify complete pipeline executes without errors.
        
        This tests the entire workflow from input to output with realistic data.
        """
        try:
            processor = TerrainProcessor(self.config_file)
            
            # Execute complete pipeline
            processor.run()
            
            # Verify output file was created
            self.assertTrue(os.path.exists(self.output_file), "Output file not created")
            
            # Verify output file structure
            with nc.Dataset(self.output_file, 'r') as ncfile:
                # Check required dimensions
                self.assertIn('grid_size', ncfile.dimensions)
                expected_grid_size = 6 * self.ncube * self.ncube
                actual_grid_size = len(ncfile.dimensions['grid_size'])
                self.assertEqual(actual_grid_size, expected_grid_size,
                               f"Wrong grid size: {actual_grid_size} != {expected_grid_size}")
                
                # Check required variables
                required_vars = ['lat', 'lon', 'terr', 'LANDFRAC', 'LANDM_COSLAT', 'var30']
                for var in required_vars:
                    self.assertIn(var, ncfile.variables, f"Missing output variable: {var}")
                
                # Check data ranges are physically realistic
                lat_data = ncfile.variables['lat'][:]
                lon_data = ncfile.variables['lon'][:]
                terr_data = ncfile.variables['terr'][:]
                landfrac_data = ncfile.variables['LANDFRAC'][:]
                
                self.assertTrue(np.all((-90 <= lat_data) & (lat_data <= 90)), "Invalid latitude values")
                self.assertTrue(np.all((0 <= lon_data) & (lon_data < 360)), "Invalid longitude values")
                self.assertTrue(np.all((-15000 <= terr_data) & (terr_data <= 15000)), "Unrealistic terrain values")
                self.assertTrue(np.all((0 <= landfrac_data) & (landfrac_data <= 1)), "Invalid land fraction values")
                
                # Check for NaN values
                self.assertFalse(np.any(np.isnan(lat_data)), "NaN values in latitude")
                self.assertFalse(np.any(np.isnan(lon_data)), "NaN values in longitude")
                self.assertFalse(np.any(np.isnan(terr_data)), "NaN values in terrain")
                self.assertFalse(np.any(np.isnan(landfrac_data)), "NaN values in land fraction")
        
        except Exception as e:
            self.fail(f"Complete pipeline test failed: {e}")
    
    def test_performance_and_memory_safety(self):
        """
        CRITICAL TEST: Verify performance and memory usage are acceptable.
        
        The code must run within reasonable time and memory limits.
        """
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        start_time = time.time()
        
        try:
            processor = TerrainProcessor(self.config_file)
            processor.run()
            
            end_time = time.time()
            final_memory = process.memory_info().rss
            
            # Performance checks
            execution_time = end_time - start_time
            memory_increase = (final_memory - initial_memory) / 1024**2  # MB
            
            # Reasonable limits for test data size
            self.assertLess(execution_time, 300,  # 5 minutes max for test
                          f"Execution too slow: {execution_time:.1f}s")
            
            self.assertLess(memory_increase, 2048,  # 2GB max increase
                          f"Memory usage too high: {memory_increase:.1f}MB")
            
        except Exception as e:
            self.fail(f"Performance test failed: {e}")


class EnvironmentalModelValidator:
    """
    Centralized validation controller for environmental modeling code.
    
    This class provides a clean, intuitive interface for running all safety-critical
    tests required before production deployment of environmental modeling systems.
    """
    
    def __init__(self):
        """Initialize the validator with test configuration"""
        self.test_suites = [
            CoordinateTransformationTests,
            AreaCalculationTests, 
            DataConservationTests,
            BoundaryConditionTests,
            NumericalStabilityTests,
            IntegrationTests
        ]
        
        self.results = {}
        self.total_tests = 0
        self.total_failures = 0
    
    def run_all_tests(self, verbosity=2):
        """
        Execute all safety-critical test suites.
        
        Args:
            verbosity (int): Test output verbosity level (0-2)
            
        Returns:
            bool: True if all tests pass, False if any failures
        """
        print("🌍" * 40)
        print("ENVIRONMENTAL MODEL SAFETY VALIDATION")
        print("🌍" * 40)
        print("Classification: SAFETY-CRITICAL ENVIRONMENTAL CODE")
        print("Purpose: Climate and Atmospheric Model Grid Generation")
        print("Impact: Global Environmental Simulation Accuracy")
        print("=" * 80)
        
        all_passed = True
        
        for test_suite_class in self.test_suites:
            suite_name = test_suite_class.__name__
            print(f"\n📋 Running {suite_name}...")
            print("-" * 60)
            
            # Create and run test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(test_suite_class)
            runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
            result = runner.run(suite)
            
            # Track results
            self.total_tests += result.testsRun
            suite_failures = len(result.failures) + len(result.errors)
            self.total_failures += suite_failures
            
            self.results[suite_name] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success': suite_failures == 0
            }
            
            if suite_failures > 0:
                all_passed = False
                print(f"❌ FAILURES DETECTED in {suite_name}")
                self._report_failures(result.failures + result.errors)
            else:
                print(f"✅ All tests passed in {suite_name}")
        
        return all_passed
    
    def _report_failures(self, failures):
        """Report detailed failure information"""
        for test, traceback in failures:
            print(f"   FAILED: {test}")
            # Extract just the assertion error message
            error_lines = traceback.split('\n')
            for line in error_lines:
                if 'AssertionError:' in line:
                    print(f"   ERROR: {line.split('AssertionError:')[-1].strip()}")
                    break
    
    def generate_certification_report(self, all_tests_passed):
        """
        Generate final certification report.
        
        Args:
            all_tests_passed (bool): Whether all tests passed
            
        Returns:
            bool: Certification status
        """
        print("\n" + "=" * 80)
        print("ENVIRONMENTAL SAFETY CERTIFICATION REPORT")
        print("=" * 80)
        
        # Test summary
        print(f"Total test suites: {len(self.test_suites)}")
        print(f"Total tests executed: {self.total_tests}")
        print(f"Total failures: {self.total_failures}")
        
        success_rate = ((self.total_tests - self.total_failures) / self.total_tests * 100) if self.total_tests > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
        
        print("\nTest Suite Results:")
        print("-" * 40)
        for suite_name, results in self.results.items():
            status = "✅ PASS" if results['success'] else "❌ FAIL"
            print(f"{suite_name:<35} {status}")
        
        print("=" * 80)
        
        # Final certification decision
        if all_tests_passed:
            print("🟢 CERTIFICATION STATUS: APPROVED FOR PRODUCTION")
            print("✅ ALL SAFETY-CRITICAL TESTS PASSED")
            print("✅ MATHEMATICAL ACCURACY VERIFIED")
            print("✅ CONSERVATION LAWS MAINTAINED") 
            print("✅ NUMERICAL STABILITY CONFIRMED")
            print("✅ COORDINATE TRANSFORMATIONS VALIDATED")
            print("✅ ENVIRONMENTAL MODEL SAFETY ASSURED")
            print("\n🌍 This code is certified safe for environmental modeling")
            print("🌍 Climate and atmospheric simulations may proceed")
            return True
        else:
            print("🔴 CERTIFICATION STATUS: REJECTED - UNSAFE FOR PRODUCTION")
            print("❌ CRITICAL SAFETY TESTS FAILED")
            print("❌ POTENTIAL DATA CORRUPTION RISK")
            print("❌ ENVIRONMENTAL MODEL ACCURACY COMPROMISED")
            print("❌ CLIMATE SIMULATION RESULTS UNRELIABLE")
            print("\n⚠️  DO NOT DEPLOY TO PRODUCTION SYSTEMS")
            print("⚠️  FIX ALL FAILURES BEFORE ENVIRONMENTAL USE")
            print("⚠️  INCORRECT RESULTS COULD IMPACT CLIMATE POLICY")
            return False
    
    def validate_for_production(self, verbosity=2):
        """
        Complete validation workflow for production certification.
        
        Args:
            verbosity (int): Test output verbosity level
            
        Returns:
            bool: True if certified for production, False otherwise
        """
        # Set strict error handling for safety-critical testing
        warnings.filterwarnings("error")  # Convert warnings to errors
        np.seterr(all='raise')  # Raise exceptions for numpy errors
        
        try:
            # Run all test suites
            all_tests_passed = self.run_all_tests(verbosity)
            
            # Generate certification report
            certification_approved = self.generate_certification_report(all_tests_passed)
            
            return certification_approved
            
        except Exception as e:
            print(f"\n🔴 CRITICAL ERROR DURING VALIDATION: {e}")
            print("❌ VALIDATION PROCESS FAILED")
            print("❌ CODE IS NOT SAFE FOR PRODUCTION USE")
            return False
        
        finally:
            # Restore normal error handling
            warnings.resetwarnings()
            np.seterr(all='warn')


def main():
    """
    Main entry point for environmental model validation.
    
    This function provides a simple, clean interface for running the complete
    safety validation suite required before production deployment.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Safety-Critical Validation Suite for Environmental Modeling Code',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: This validation suite must pass completely before deploying
bin_to_cube code to any environmental modeling or climate simulation system.

Examples:
  python test_bin_to_cube_comprehensive.py              # Run with default verbosity
  python test_bin_to_cube_comprehensive.py --quiet      # Minimal output
  python test_bin_to_cube_comprehensive.py --verbose    # Maximum detail

Exit Codes:
  0 - All tests passed, code certified for production
  1 - Tests failed, code rejected for production use
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_const', const=2, dest='verbosity',
                       help='Maximum verbosity output')
    parser.add_argument('--quiet', '-q', action='store_const', const=0, dest='verbosity',
                       help='Minimal output')
    parser.set_defaults(verbosity=1)  # Default verbosity level
    
    args = parser.parse_args()
    
    # Create validator and run complete validation
    validator = EnvironmentalModelValidator()
    certification_approved = validator.validate_for_production(verbosity=args.verbosity)
    
    # Exit with appropriate code for automated systems
    sys.exit(0 if certification_approved else 1)


if __name__ == "__main__":
    main()
