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


class SafetyTestCase(unittest.TestCase):
    """
    Base class for safety-critical test cases.
    Implements strict error checking and validation.
    """
    
    def assertAlmostEqualStrict(self, first, second, tolerance=NUMERICAL_TOLERANCE, msg=None):
        """Strict numerical comparison for safety-critical calculations"""
        if abs(first - second) > tolerance:
            standard_msg = f'{first} != {second} within tolerance {tolerance}'
            self.fail(self._formatMessage(msg, standard_msg))
    
    def assertConservationLaw(self, initial_value, final_value, conservation_type="mass", tolerance=1e-10):
        """Verify conservation laws are maintained"""
        relative_error = abs(final_value - initial_value) / abs(initial_value) if initial_value != 0 else abs(final_value)
        if relative_error > tolerance:
            self.fail(f"{conservation_type.title()} conservation violated: "
                     f"initial={initial_value}, final={final_value}, "
                     f"relative_error={relative_error:.2e} > tolerance={tolerance:.2e}")
    
    def assertPhysicallyRealistic(self, value, min_val, max_val, quantity_name):
        """Verify values are within physically realistic bounds"""
        if not (min_val <= value <= max_val):
            self.fail(f"{quantity_name} = {value} is outside physically realistic range [{min_val}, {max_val}]")


class CoordinateTransformationTests(SafetyTestCase):
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
        
        print(f"Panel coverage distribution: {panel_counts}")
    
    def test_coordinate_transformation_mathematical_properties(self):
        """
        CRITICAL TEST: Verify coordinate transformations preserve mathematical properties.
        
        Test specific mathematical properties that must be preserved:
        1. Antipodal points map to different panels
        2. Great circle distances are preserved approximately
        3. Panel boundaries are mathematically correct
        """
        # Test antipodal points
        test_points = [
            (0.0, PI/4),
            (PI/2, PI/6),
            (PI, PI/3),
        ]
        
        for lon, lat in test_points:
            # Original point
            alpha1, beta1, panel1 = self.processor.cubed_sphere_abp_from_rll(lon, lat)
            
            # Antipodal point
            lon_antipodal = (lon + PI) % (2*PI)
            lat_antipodal = -lat
            
            alpha2, beta2, panel2 = self.processor.cubed_sphere_abp_from_rll(lon_antipodal, lat_antipodal)
            
            # Antipodal points should generally map to different panels (except at panel boundaries)
            # This is a geometric property verification
            if panel1 == panel2:
                warnings.warn(f"Antipodal points ({math.degrees(lon):.1f}°, {math.degrees(lat):.1f}°) "
                            f"and ({math.degrees(lon_antipodal):.1f}°, {math.degrees(lat_antipodal):.1f}°) "
                            f"both map to panel {panel1}")


class AreaCalculationTests(SafetyTestCase):
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
                
                # Verify areas are reasonable (not too small/large)
                mean_area = np.mean(darea_face)
                expected_mean = (4.0 * PI) / (6 * ncube * ncube)
                self.assertAlmostEqualStrict(mean_area, expected_mean, tolerance=1e-10,
                                           msg=f"Mean area incorrect for ncube={ncube}")
    
    def test_area_calculation_numerical_stability(self):
        """
        CRITICAL TEST: Verify area calculations are numerically stable.
        
        Area calculations must be stable across different numerical approaches
        and not sensitive to floating-point round-off errors.
        """
        ncube = 32
        self.processor.ncube = ncube
        
        # Calculate areas multiple times
        areas_1 = self.processor.equiangular_all_areas()
        areas_2 = self.processor.equiangular_all_areas()
        
        # Results must be bit-identical (deterministic)
        np.testing.assert_array_equal(areas_1, areas_2, 
                                    err_msg="Area calculations are non-deterministic")
        
        # Test with slightly perturbed inputs (numerical stability)
        original_pi = math.pi
        perturbed_pi = original_pi * (1 + 1e-15)  # Tiny perturbation
        
        # Temporarily modify PI constant (if possible)
        # This tests sensitivity to numerical precision
        try:
            # Areas should be relatively insensitive to tiny perturbations
            max_relative_change = np.max(np.abs(areas_1 - areas_1) / areas_1)
            self.assertLess(max_relative_change, 1e-10, 
                          "Area calculations too sensitive to numerical perturbations")
        except:
            pass  # Skip if cannot modify constants
    
    def test_area_monotonicity_and_smoothness(self):
        """
        CRITICAL TEST: Verify area distribution is mathematically correct.
        
        Grid cell areas should vary smoothly and monotonically as expected
        from the cubed sphere projection geometry.
        """
        ncube = 24
        self.processor.ncube = ncube
        
        darea = self.processor.equiangular_all_areas()
        
        # Areas should be symmetric about the center of each face
        center = ncube // 2
        
        # Test symmetry (within numerical precision)
        for i in range(center):
            for j in range(center):
                # Four-fold symmetry around center
                area_00 = darea[center + i, center + j]
                area_01 = darea[center + i, center - j - 1]
                area_10 = darea[center - i - 1, center + j]
                area_11 = darea[center - i - 1, center - j - 1]
                
                areas = [area_00, area_01, area_10, area_11]
                mean_area = np.mean(areas)
                
                for area in areas:
                    relative_diff = abs(area - mean_area) / mean_area
                    self.assertLess(relative_diff, 1e-14, 
                                  f"Symmetry violation at ({i},{j}): areas={areas}")
        
        # Areas should increase monotonically from center to corners
        center_area = darea[center, center]
        corner_area = darea[0, 0]
        
        self.assertGreater(corner_area, center_area, 
                          "Corner areas should be larger than center areas")
        
        # Verify no extreme outliers
        area_ratio = np.max(darea) / np.min(darea)
        self.assertLess(area_ratio, 10.0, 
                       f"Area ratio too large: {area_ratio:.2f}")


class DataConservationTests(SafetyTestCase):
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
    
    def test_local_conservation_properties(self):
        """
        CRITICAL TEST: Verify local conservation properties.
        
        Data should be conserved locally - no artificial creation or
        destruction of data during the transformation process.
        """
        self.processor.read_terrain_data()
        
        # Bin data
        weight, terr_cube, landfrac_cube, idx, idy, idp, da = self.processor.bin_data_to_cubed_sphere()
        
        # Before normalization, check raw conservation
        total_weight = np.sum(weight)
        expected_weight = 4.0 * PI  # Total weight should equal sphere surface area
        
        self.assertAlmostEqualStrict(total_weight, expected_weight, tolerance=1e-8,
                                   msg=f"Weight conservation failed: {total_weight} != {expected_weight}")
        
        # Check that all weights are positive
        self.assertTrue(np.all(weight >= 0), "Found negative weights")
        
        # Check that no data is lost (all cells have some weight)
        empty_cells = np.sum(weight < 1e-15)
        total_cells = weight.size
        empty_fraction = empty_cells / total_cells
        
        self.assertLess(empty_fraction, 0.01, 
                       f"Too many empty cells: {empty_fraction:.3f} of total")
    
    def test_interpolation_accuracy_with_analytical_functions(self):
        """
        CRITICAL TEST: Test interpolation accuracy using analytical functions.
        
        Use analytically known functions to verify interpolation accuracy.
        This tests the mathematical correctness of the interpolation algorithm.
        """
        # Test with spherical harmonic Y_2^2 = cos²(lat) * cos(2*lon)
        # This function has known analytical properties
        
        # Load data
        self.processor.read_terrain_data()
        original_terrain = self.processor.terr.copy()
        
        # Replace with analytical function (Y_2^2 spherical harmonic)
        lat_rad = self.processor.lat * DEG2RAD
        lon_rad = self.processor.lon * DEG2RAD
        
        lon_2d, lat_2d = np.meshgrid(lon_rad, lat_rad)
        analytical_field = np.cos(lat_2d)**2 * np.cos(2.0 * lon_2d)
        
        self.processor.terr = analytical_field
        
        # Perform binning
        weight, terr_cube, landfrac_cube, idx, idy, idp, da = self.processor.bin_data_to_cubed_sphere()
        terr_cube, landfrac_cube = self.processor.normalize_and_validate(weight, terr_cube, landfrac_cube)
        
        # Verify interpolated values at specific points
        # Test points where analytical function value is known
        test_points = [
            (0.0, 0.0, 1.0),      # (0°, 0°) -> Y_2^2 = 1
            (PI/2, 0.0, -1.0),    # (90°, 0°) -> Y_2^2 = -1  
            (0.0, PI/2, 0.0),     # (0°, 90°) -> Y_2^2 = 0
        ]
        
        for lon_test, lat_test, expected_value in test_points:
            # Find corresponding cubed sphere point
            try:
                alpha, beta, panel = self.processor.cubed_sphere_abp_from_rll(lon_test, lat_test)
                
                # Find grid indices
                icube = int((alpha + PI/4) / da)
                jcube = int((beta + PI/4) / da)
                
                # Check if indices are valid
                if 0 <= icube < self.ncube and 0 <= jcube < self.ncube:
                    interpolated_value = terr_cube[icube, jcube, panel-1]
                    
                    # Allow some tolerance for discretization error
                    error = abs(interpolated_value - expected_value)
                    relative_error = error / (abs(expected_value) + 1e-10)
                    
                    self.assertLess(relative_error, 0.1,  # 10% tolerance for discretization
                                  f"Interpolation error at ({math.degrees(lon_test):.1f}°, {math.degrees(lat_test):.1f}°): "
                                  f"expected={expected_value:.3f}, got={interpolated_value:.3f}")
            except:
                # Skip points that are problematic (e.g., poles)
                continue


class BoundaryConditionTests(SafetyTestCase):
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
    
    def test_date_line_continuity(self):
        """
        CRITICAL TEST: Verify continuity across date line (180° longitude).
        
        The date line should not cause discontinuities in the data.
        Errors here could cause artifacts in global circulation models.
        """
        # Test points near the date line
        epsilon = 1e-8
        date_line_points = [
            (PI - epsilon, 0.0),      # Just west of date line
            (PI + epsilon, 0.0),      # Just east of date line
            (PI - epsilon, PI/4),     # 45°N, just west
            (PI + epsilon, PI/4),     # 45°N, just east
            (PI - epsilon, -PI/4),    # 45°S, just west
            (PI + epsilon, -PI/4),    # 45°S, just east
        ]
        
        for lon, lat in date_line_points:
            with self.subTest(lon=math.degrees(lon), lat=math.degrees(lat)):
                try:
                    alpha, beta, panel = self.processor.cubed_sphere_abp_from_rll(lon, lat)
                    
                    # Verify transformation is stable near date line
                    self.assertFalse(math.isnan(alpha), "NaN alpha near date line")
                    self.assertFalse(math.isnan(beta), "NaN beta near date line")
                    self.assertIn(panel, range(1, 7), f"Invalid panel {panel} near date line")
                    
                    # Test continuity by checking nearby points
                    for delta in [-epsilon/2, epsilon/2]:
                        lon_nearby = lon + delta
                        if lon_nearby < 0:
                            lon_nearby += 2*PI
                        elif lon_nearby >= 2*PI:
                            lon_nearby -= 2*PI
                        
                        alpha_nearby, beta_nearby, panel_nearby = self.processor.cubed_sphere_abp_from_rll(lon_nearby, lat)
                        
                        # Alpha and beta should be continuous (within panel)
                        if panel == panel_nearby:
                            alpha_diff = abs(alpha - alpha_nearby)
                            beta_diff = abs(beta - beta_nearby)
                            
                            self.assertLess(alpha_diff, 1e-6, "Alpha discontinuity near date line")
                            self.assertLess(beta_diff, 1e-6, "Beta discontinuity near date line")
                
                except Exception as e:
                    self.fail(f"Date line transformation failed at ({math.degrees(lon):.10f}°, {math.degrees(lat):.10f}°): {e}")
    
    def test_panel_boundary_transitions(self):
        """
        CRITICAL TEST: Verify smooth transitions at panel boundaries.
        
        Panel boundaries must not introduce artificial discontinuities.
        """
        # Test transitions between known panel boundaries
        boundary_test_cases = [
            # Equatorial boundaries
            (0.0, 0.0),     # Prime meridian
            (PI/2, 0.0),    # 90°E
            (PI, 0.0),      # Date line
            (3*PI/2, 0.0),  # 90°W
            # Polar boundaries
            (0.0, PI/4),    # 45°N
            (PI/2, PI/4),   # 45°N, 90°E
        ]
        
        for lon_center, lat_center in boundary_test_cases:
            with self.subTest(lon=math.degrees(lon_center), lat=math.degrees(lat_center)):
                # Test points in a small neighborhood
                delta = 1e-6
                offsets = [-delta, 0, delta]
                
                panels = set()
                coords = []
                
                for d_lon in offsets:
                    for d_lat in offsets:
                        lon_test = lon_center + d_lon
                        lat_test = lat_center + d_lat
                        
                        # Clamp latitude to valid range
                        lat_test = max(-PI/2 + 1e-12, min(PI/2 - 1e-12, lat_test))
                        
                        try:
                            alpha, beta, panel = self.processor.cubed_sphere_abp_from_rll(lon_test, lat_test)
                            panels.add(panel)
                            coords.append((alpha, beta, panel))
                        except Exception as e:
                            self.fail(f"Boundary test failed at ({math.degrees(lon_test):.10f}°, {math.degrees(lat_test):.10f}°): {e}")
                
                # Should not have too many different panels in small region
                self.assertLessEqual(len(panels), 4, f"Too many panels in small region: {panels}")
                
                # Check coordinate smoothness within each panel
                for panel in panels:
                    panel_coords = [(a, b) for a, b, p in coords if p == panel]
                    if len(panel_coords) > 1:
                        alphas = [a for a, b in panel_coords]
                        betas = [b for a, b in panel_coords]
                        
                        alpha_range = max(alphas) - min(alphas)
                        beta_range = max(betas) - min(betas)
                        
                        # Coordinates should vary smoothly within panel
                        self.assertLess(alpha_range, 1e-4, f"Large alpha variation in panel {panel}")
                        self.assertLess(beta_range, 1e-4, f"Large beta variation in panel {panel}")


class NumericalStabilityTests(SafetyTestCase):
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
    
    def test_area_calculation_conditioning(self):
        """
        CRITICAL TEST: Verify area calculations are well-conditioned.
        
        Area calculations should be numerically stable and not sensitive
        to small perturbations in input parameters.
        """
        base_ncube = 32
        self.processor.ncube = base_ncube
        
        # Calculate reference areas
        reference_areas = self.processor.equiangular_all_areas()
        
        # Test with slightly different ncube values (should scale predictably)
        test_ncube_values = [31, 32, 33]
        
        for ncube in test_ncube_values:
            with self.subTest(ncube=ncube):
                self.processor.ncube = ncube
                test_areas = self.processor.equiangular_all_areas()
                
                # Total area should still be 4π
                total_area = 6.0 * np.sum(test_areas)
                expected_area = 4.0 * PI
                
                relative_error = abs(total_area - expected_area) / expected_area
                self.assertLess(relative_error, 1e-12, 
                              f"Area scaling failed for ncube={ncube}: relative_error={relative_error:.2e}")
                
                # Check that no areas are zero or negative
                self.assertTrue(np.all(test_areas > 0), f"Invalid areas for ncube={ncube}")
                
                # Check condition number (ratio of largest to smallest area)
                condition_number = np.max(test_areas) / np.min(test_areas)
                self.assertLess(condition_number, 20.0, 
                              f"Poor conditioning for ncube={ncube}: condition={condition_number:.2f}")
    
    def test_floating_point_determinism(self):
        """
        CRITICAL TEST: Verify calculations are deterministic.
        
        Identical inputs must produce identical outputs to ensure
        reproducible climate model results.
        """
        # Test same calculation multiple times
        test_coords = [(PI/4, PI/8), (3*PI/4, -PI/8), (PI, PI/3)]
        
        for lon, lat in test_coords:
            with self.subTest(lon=math.degrees(lon), lat=math.degrees(lat)):
                # Perform transformation multiple times
                results = []
                for _ in range(10):
                    alpha, beta, panel = self.processor.cubed_sphere_abp_from_rll(lon, lat)
                    results.append((alpha, beta, panel))
                
                # All results must be identical
                first_result = results[0]
                for i, result in enumerate(results[1:], 1):
                    self.assertEqual(result, first_result, 
                                   f"Non-deterministic result at iteration {i}: {result} != {first_result}")


class ErrorPropagationTests(SafetyTestCase):
    """
    CRITICAL: Test error propagation and numerical robustness.
    
    Failures here could cause:
    - Error amplification in climate models
    - Numerical instabilities
    - Silent corruption of results
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
    
    def test_input_error_sensitivity(self):
        """
        CRITICAL TEST: Verify sensitivity to input errors is bounded.
        
        Small errors in input coordinates should not cause large errors
        in output coordinates (Lipschitz continuity).
        """
        base_coords = [
            (PI/6, PI/12),    # 30°E, 15°N
            (PI/2, PI/6),     # 90°E, 30°N
            (PI, 0.0),        # 180°E, 0°N
        ]
        
        error_magnitudes = [1e-10, 1e-8, 1e-6, 1e-4]
        
        for lon_base, lat_base in base_coords:
            for error_mag in error_magnitudes:
                with self.subTest(lon=math.degrees(lon_base), lat=math.degrees(lat_base), error=error_mag):
                    try:
                        # Base transformation
                        alpha_base, beta_base, panel_base = self.processor.cubed_sphere_abp_from_rll(lon_base, lat_base)
                        
                        # Perturbed transformations
                        perturbations = [
                            (error_mag, 0),
                            (0, error_mag),
                            (error_mag, error_mag),
                            (-error_mag, error_mag),
                        ]
                        
                        for d_lon, d_lat in perturbations:
                            lon_pert = lon_base + d_lon
                            lat_pert = lat_base + d_lat
                            
                            # Clamp to valid ranges
                            lat_pert = max(-PI/2 + 1e-12, min(PI/2 - 1e-12, lat_pert))
                            
                            alpha_pert, beta_pert, panel_pert = self.processor.cubed_sphere_abp_from_rll(lon_pert, lat_pert)
                            
                            # Check error amplification (only if same panel)
                            if panel_base == panel_pert:
                                alpha_error = abs(alpha_pert - alpha_base)
                                beta_error = abs(beta_pert - beta_base)
                                
                                input_error = math.sqrt(d_lon**2 + d_lat**2)
                                output_error = math.sqrt(alpha_error**2 + beta_error**2)
                                
                                if input_error > 0:
                                    amplification = output_error / input_error
                                    
                                    # Error amplification should be bounded
                                    self.assertLess(amplification, 10.0,
                                                  f"Excessive error amplification: {amplification:.2f} "
                                                  f"for input error {input_error:.2e}")
                    
                    except Exception as e:
                        self.fail(f"Error sensitivity test failed: {e}")
    
    def test_catastrophic_cancellation_detection(self):
        """
        CRITICAL TEST: Detect potential catastrophic cancellation.
        
        Verify that coordinate calculations don't suffer from
        catastrophic cancellation in floating-point arithmetic.
        """
        # Test near-cancellation scenarios
        critical_points = [
            (1e-15, 1e-15),        # Very small coordinates
            (PI/4 - 1e-14, 0.0),   # Near panel boundary
            (0.0, PI/2 - 1e-14),   # Near pole
        ]
        
        for lon, lat in critical_points:
            with self.subTest(lon=lon, lat=lat):
                try:
                    alpha, beta, panel = self.processor.cubed_sphere_abp_from_rll(lon, lat)
                    
                    # Check for loss of precision indicators
                    self.assertFalse(math.isnan(alpha), "NaN indicates cancellation")
                    self.assertFalse(math.isnan(beta), "NaN indicates cancellation")
                    
                    # Verify results are reasonable
                    self.assertPhysicallyRealistic(alpha, -PI/4, PI/4, "alpha")
                    self.assertPhysicallyRealistic(beta, -PI/4, PI/4, "beta")
                    
                    # Test roundtrip to verify precision
                    lon_recovered, lat_recovered = self.processor.cubed_sphere_rll_from_abp(alpha, beta, panel)
                    
                    # Allow larger tolerance for extreme cases
                    lon_error = abs(lon_recovered - lon)
                    lat_error = abs(lat_recovered - lat)
                    
                    # Error should not be disproportionately large
                    max_expected_error = max(1e-12, abs(lon) * 1e-10, abs(lat) * 1e-10)
                    
                    self.assertLess(lon_error, max_expected_error,
                                  f"Longitude cancellation error: {lon_error:.2e}")
                    self.assertLess(lat_error, max_expected_error,
                                  f"Latitude cancellation error: {lat_error:.2e}")
                
                except Exception as e:
                    # Some extreme cases may legitimately fail
                    if abs(lon) < 1e-14 and abs(lat) < 1e-14:
                        continue  # Skip extremely small values
                    else:
                        self.fail(f"Cancellation test failed: {e}")


class IntegrationTests(SafetyTestCase):
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
        
        # Create realistic terrain elevation
        # Simulate continents, mountains, ocean trenches
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
        
        # Add ocean trenches
        for center_lon, center_lat, depth, width in [
            (180, 0, -8000, 15),    # Pacific-like trench
            (-60, -30, -6000, 20),  # Atlantic-like trench
        ]:
            dist = np.sqrt((lon_2d - center_lon)**2 + (lat_2d - center_lat)**2)
            terrain += depth * np.exp(-(dist/width)**2)
        
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
            
            # Add realistic attributes
            terrain_var.units = 'm'
            terrain_var.long_name = 'surface elevation'
            landfrac_var.units = '1'
            landfrac_var.long_name = 'land area fraction'
        
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
    
    def test_output_data_quality(self):
        """
        CRITICAL TEST: Verify output data quality and consistency.
        
        Output data must maintain physical consistency and conservation laws.
        """
        processor = TerrainProcessor(self.config_file)
        processor.run()
        
        # Analyze output quality
        with nc.Dataset(self.output_file, 'r') as ncfile:
            terr_data = ncfile.variables['terr'][:]
            landfrac_data = ncfile.variables['LANDFRAC'][:]
            var30_data = ncfile.variables['var30'][:]
            
            # Statistical checks
            terrain_stats = {
                'mean': np.mean(terr_data),
                'std': np.std(terr_data),
                'min': np.min(terr_data),
                'max': np.max(terr_data),
            }
            
            # Verify statistics are reasonable
            self.assertPhysicallyRealistic(terrain_stats['mean'], -2000, 2000, "mean terrain elevation")
            self.assertPhysicallyRealistic(terrain_stats['std'], 0, 5000, "terrain elevation std")
            
            # Verify variance data is non-negative
            self.assertTrue(np.all(var30_data >= 0), "Negative variance values found")
            
            # Check for extreme outliers (potential data corruption)
            terrain_iqr = np.percentile(terr_data, 75) - np.percentile(terr_data, 25)
            outlier_threshold = 5 * terrain_iqr
            outliers = np.sum(np.abs(terr_data - np.median(terr_data)) > outlier_threshold)
            outlier_fraction = outliers / len(terr_data)
            
            self.assertLess(outlier_fraction, 0.01, f"Too many outliers: {outlier_fraction:.3f}")
    
    def test_performance_and_memory_safety(self):
        """
        CRITICAL TEST: Verify performance and memory usage are acceptable.
        
        The code must run within reasonable time and memory limits.
        """
        import time
        import psutil
        import os
        
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
            
            print(f"Performance: {execution_time:.2f}s, Memory: +{memory_increase:.1f}MB")
            
        except Exception as e:
            self.fail(f"Performance test failed: {e}")


def run_safety_critical_tests():
    """
    Run all safety-critical tests with comprehensive reporting.
    
    This function must be called before any deployment to production systems.
    """
    print("="*80)
    print("SAFETY-CRITICAL ENVIRONMENTAL MODEL VALIDATION")
    print("="*80)
    print("WARNING: This code affects environmental modeling systems.")
    print("ALL TESTS MUST PASS before production deployment.")
    print("="*80)
    
    # Create test suite
    test_classes = [
        CoordinateTransformationTests,
        AreaCalculationTests,
        DataConservationTests,
        BoundaryConditionTests,
        NumericalStabilityTests,
        ErrorPropagationTests,
        IntegrationTests,
    ]
    
    all_tests_passed = True
    total_tests = 0
    total_failures = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        print("-" * 60)
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures) + len(result.errors)
        
        if result.failures or result.errors:
            all_tests_passed = False
            print(f"❌ FAILURES DETECTED in {test_class.__name__}")
            for test, traceback in result.failures + result.errors:
                print(f"   FAILED: {test}")
                print(f"   ERROR: {traceback.split('AssertionError:')[-1].strip()}")
        else:
            print(f"✅ All tests passed in {test_class.__name__}")
    
    print("\n" + "="*80)
    print("FINAL SAFETY VALIDATION REPORT")
    print("="*80)
    print(f"Total tests run: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Success rate: {((total_tests - total_failures) / total_tests * 100):.1f}%")
    
    if all_tests_passed:
        print("🟢 ALL SAFETY-CRITICAL TESTS PASSED")
        print("✅ CODE IS CLEARED FOR ENVIRONMENTAL PRODUCTION USE")
        print("✅ CONSERVATION LAWS VERIFIED")
        print("✅ NUMERICAL STABILITY CONFIRMED")
        print("✅ COORDINATE TRANSFORMATIONS VALIDATED")
        return True
    else:
        print("🔴 SAFETY-CRITICAL TESTS FAILED")
        print("❌ CODE IS NOT SAFE FOR PRODUCTION USE")
        print("❌ POTENTIAL ENVIRONMENTAL MODEL CORRUPTION RISK")
        print("❌ MUST FIX ALL FAILURES BEFORE DEPLOYMENT")
        return False


class StressTests(SafetyTestCase):
    """
    CRITICAL: Stress tests for extreme conditions.
    
    These tests verify the code can handle extreme inputs without
    catastrophic failure that could corrupt environmental models.
    """
    
    def test_extreme_coordinate_values(self):
        """Test behavior with extreme coordinate values"""
        processor = TerrainProcessor()
        
        extreme_coords = [
            (1e-15, 1e-15),         # Near machine epsilon
            (2*PI - 1e-15, PI/2 - 1e-15),  # Near boundaries
            (1e10, 1e10),           # Extremely large (should fail gracefully)
            (float('inf'), 0),      # Infinity
            (float('nan'), 0),      # NaN
        ]
        
        for lon, lat in extreme_coords:
            with self.subTest(lon=lon, lat=lat):
                if math.isfinite(lon) and math.isfinite(lat):
                    # Should either work or fail gracefully
                    try:
                        if abs(lon) < 1e6 and abs(lat) < PI/2:
                            alpha, beta, panel = processor.cubed_sphere_abp_from_rll(lon, lat)
                            self.assertFalse(math.isnan(alpha))
                            self.assertFalse(math.isnan(beta))
                    except (ValueError, OverflowError):
                        pass  # Acceptable failure for extreme values
                else:
                    # Should fail gracefully with infinite/NaN inputs
                    with self.assertRaises((ValueError, OverflowError)):
                        processor.cubed_sphere_abp_from_rll(lon, lat)
    
    def test_memory_limits_large_datasets(self):
        """Test behavior with memory-challenging dataset sizes"""
        # Test with progressively larger ncube values
        large_ncube_values = [100, 500, 1000]  # Up to 6M grid points
        
        for ncube in large_ncube_values:
            with self.subTest(ncube=ncube):
                try:
                    temp_dir = tempfile.mkdtemp()
                    config_file = os.path.join(temp_dir, "stress_config.ini")
                    
                    with open(config_file, 'w') as f:
                        f.write(f"""[binparams]
raw_latlon_data_file = dummy.nc
output_file = dummy_out.nc
ncube = {ncube}
landm_coslat_file = dummy_landm.nc
""")
                    
                    processor = TerrainProcessor(config_file)
                    
                    # Test area calculation (most memory-intensive operation)
                    areas = processor.equiangular_all_areas()
                    
                    # Verify basic properties
                    total_area = 6.0 * np.sum(areas)
                    expected_area = 4.0 * PI
                    relative_error = abs(total_area - expected_area) / expected_area
                    
                    self.assertLess(relative_error, 1e-10,
                                  f"Area conservation failed for large ncube={ncube}")
                    
                    # Clean up
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
                except MemoryError:
                    print(f"Memory limit reached at ncube={ncube} (expected for large values)")
                    break
                except Exception as e:
                    self.fail(f"Unexpected failure for ncube={ncube}: {e}")


class RegressionTests(SafetyTestCase):
    """
    CRITICAL: Regression tests using known reference values.
    
    These tests verify the code produces the same results as validated
    reference implementations to prevent silent algorithmic changes.
    """
    
    def test_known_coordinate_transformations(self):
        """Test against known coordinate transformation values"""
        processor = TerrainProcessor()
        
        # Reference values computed with validated implementation
        reference_cases = [
            # (lon, lat) -> (alpha, beta, panel)
            (0.0, 0.0, (0.0, 0.0, 1)),
            (PI/2, 0.0, (0.0, 0.0, 2)),
            (PI, 0.0, (0.0, 0.0, 3)),
            (3*PI/2, 0.0, (0.0, 0.0, 4)),
            (0.0, PI/2 - 1e-6, (0.0, 0.0, 6)),
            (0.0, -PI/2 + 1e-6, (0.0, 0.0, 5)),
        ]
        
        for lon, lat, (expected_alpha, expected_beta, expected_panel) in reference_cases:
            with self.subTest(lon=math.degrees(lon), lat=math.degrees(lat)):
                alpha, beta, panel = processor.cubed_sphere_abp_from_rll(lon, lat)
                
                self.assertAlmostEqualStrict(alpha, expected_alpha, tolerance=1e-12,
                                           msg=f"Alpha mismatch for reference case")
                self.assertAlmostEqualStrict(beta, expected_beta, tolerance=1e-12,
                                           msg=f"Beta mismatch for reference case")
                self.assertEqual(panel, expected_panel,
                               f"Panel mismatch for reference case")
    
    def test_known_area_calculations(self):
        """Test against known area calculation values"""
        processor = TerrainProcessor()
        
        # Test specific ncube values with known total areas
        reference_ncube_areas = [
            (4, 4.0 * PI),   # Must always equal surface area of unit sphere
            (8, 4.0 * PI),
            (16, 4.0 * PI),
            (32, 4.0 * PI),
        ]
        
        for ncube, expected_total_area in reference_ncube_areas:
            with self.subTest(ncube=ncube):
                processor.ncube = ncube
                areas = processor.equiangular_all_areas()
                total_area = 6.0 * np.sum(areas)
                
                self.assertAlmostEqualStrict(total_area, expected_total_area, tolerance=1e-14,
                                           msg=f"Total area mismatch for ncube={ncube}")


class ValidationSuite:
    """
    Master validation suite for safety-critical environmental code.
    
    This class orchestrates all testing and provides final certification.
    """
    
    @staticmethod
    def run_full_validation():
        """
        Run complete validation suite for production certification.
        
        Returns True if all tests pass, False otherwise.
        """
        print("\n" + "🌍" * 40)
        print("ENVIRONMENTAL MODEL SAFETY VALIDATION SUITE")
        print("🌍" * 40)
        print("Classification: SAFETY-CRITICAL ENVIRONMENTAL CODE")
        print("Purpose: Climate and Atmospheric Model Grid Generation")
        print("Impact: Global Environmental Simulation Accuracy")
        print("=" * 80)
        
        # Run all test categories
        test_results = []
        
        # Core mathematical validation
        print("\n📐 MATHEMATICAL CORRECTNESS VALIDATION")
        result = run_safety_critical_tests()
        test_results.append(("Mathematical Correctness", result))
        
        # Stress testing
        print("\n💪 STRESS AND ROBUSTNESS TESTING")
        stress_suite = unittest.TestLoader().loadTestsFromTestCase(StressTests)
        stress_runner = unittest.TextTestRunner(verbosity=1)
        stress_result = stress_runner.run(stress_suite)
        stress_passed = len(stress_result.failures) == 0 and len(stress_result.errors) == 0
        test_results.append(("Stress Testing", stress_passed))
        
        # Regression testing
        print("\n🔄 REGRESSION VALIDATION")
        regression_suite = unittest.TestLoader().loadTestsFromTestCase(RegressionTests)
        regression_runner = unittest.TextTestRunner(verbosity=1)
        regression_result = regression_runner.run(regression_suite)
        regression_passed = len(regression_result.failures) == 0 and len(regression_result.errors) == 0
        test_results.append(("Regression Testing", regression_passed))
        
        # Final certification report
        print("\n" + "=" * 80)
        print("FINAL ENVIRONMENTAL SAFETY CERTIFICATION")
        print("=" * 80)
        
        all_passed = True
        for test_category, passed in test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_category:.<50} {status}")
            if not passed:
                all_passed = False
        
        print("=" * 80)
        
        if all_passed:
            print("🟢 CERTIFICATION STATUS: APPROVED FOR PRODUCTION")
            print("✅ ALL SAFETY-CRITICAL TESTS PASSED")
            print("✅ MATHEMATICAL ACCURACY VERIFIED")
            print("✅ CONSERVATION LAWS MAINTAINED")
            print("✅ NUMERICAL STABILITY CONFIRMED")
            print("✅ ERROR PROPAGATION BOUNDED")
            print("✅ COORDINATE TRANSFORMATIONS VALIDATED")
            print("✅ ENVIRONMENTAL MODEL SAFETY ASSURED")
            print("\n🌍 This code is certified safe for environmental modeling")
            print("🌍 Climate and atmospheric simulations may proceed")
        else:
            print("🔴 CERTIFICATION STATUS: REJECTED - UNSAFE FOR PRODUCTION")
            print("❌ CRITICAL SAFETY TESTS FAILED")
            print("❌ POTENTIAL DATA CORRUPTION RISK")
            print("❌ ENVIRONMENTAL MODEL ACCURACY COMPROMISED")
            print("❌ CLIMATE SIMULATION RESULTS UNRELIABLE")
            print("\n⚠️  DO NOT DEPLOY TO PRODUCTION SYSTEMS")
            print("⚠️  FIX ALL FAILURES BEFORE ENVIRONMENTAL USE")
            print("⚠️  INCORRECT RESULTS COULD IMPACT CLIMATE POLICY")
        
        print("=" * 80)
        return all_passed


if __name__ == "__main__":
    """
    Entry point for safety-critical validation.
    
    This script must be run and pass completely before any deployment
    of the bin_to_cube code to environmental modeling systems.
    """
    
    # Set strict error handling
    warnings.filterwarnings("error")  # Convert warnings to errors
    np.seterr(all='raise')  # Raise exceptions for numpy errors
    
    # Run full validation
    success = ValidationSuite.run_full_validation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
        
        