#!/usr/bin/env python3
"""
Python conversion of bin_to_cube.F90
Reads lat-lon terrain dataset from NetCDF file and bins it to cubed-sphere grid
"""

import numpy as np
import netCDF4 as nc
import math
import sys
from datetime import datetime

# Constants
PI = math.pi
PIQ = 0.25 * PI
RAD2DEG = 180.0 / PI
DEG2RAD = PI / 180.0
TINY = 1.0e-10

def read_config():
    """Read configuration (simplified - normally would read namelist)"""
    # In real version, would read from bin_to_cube.nl file
    config = {
        'raw_latlon_data_file': 'input_terrain.nc',
        'output_file': 'output_cubed_sphere.nc',
        'ncube': 3000  # cubed-sphere resolution
    }
    return config

def handle_err(status, message="NetCDF error"):
    """Error handler"""
    if status != 0:
        print(f"Error: {message}")
        sys.exit(1)

def cubed_sphere_abp_from_rll(lon, lat):
    """
    Convert lat-lon to cubed sphere (alpha, beta, panel) coordinates
    """
    rotate_cube = 0.0
    
    # Convert to Cartesian coordinates
    xx = math.cos(lon - rotate_cube) * math.cos(lat)
    yy = math.sin(lon - rotate_cube) * math.cos(lat)
    zz = math.sin(lat)
    
    pm = max(abs(xx), abs(yy), abs(zz))
    
    # Check which coordinate is maximum
    ix = 1 if pm == abs(xx) and xx > 0 else (-1 if pm == abs(xx) and xx < 0 else 0)
    iy = 1 if pm == abs(yy) and yy > 0 else (-1 if pm == abs(yy) and yy < 0 else 0)
    iz = 1 if pm == abs(zz) and zz > 0 else (-1 if pm == abs(zz) and zz < 0 else 0)
    
    # Panel assignments
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
        raise ValueError(f"CubedSphereABPFromRLL failed: ({xx}, {yy}, {zz})")
    
    alpha = math.atan(sx / sz)
    beta = math.atan(sy / sz)
    
    return alpha, beta, ipanel

def cubed_sphere_xyz_from_abp(alpha, beta, ipanel):
    """Convert cubed sphere coordinates to Cartesian"""
    a1 = math.tan(alpha)
    b1 = math.tan(beta)
    
    sz = (1.0 + a1*a1 + b1*b1)**(-0.5)
    sx = sz * a1
    sy = sz * b1
    
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
        raise ValueError(f"Panel out of range: {ipanel}")
    
    return xx, yy, zz

def cubed_sphere_rll_from_abp(alpha, beta, ipanel):
    """Convert cubed sphere coordinates back to lat-lon"""
    xx, yy, zz = cubed_sphere_xyz_from_abp(alpha, beta, ipanel)
    
    lat = math.asin(zz)
    if xx == 0.0 and yy == 0.0:
        lon = 0.0
    else:
        lon = math.atan2(yy, xx)
        if lon < 0.0:
            lon += 2.0 * PI
        if lon > 2.0 * PI:
            lon -= 2.0 * PI
    
    return lon, lat

def equiangular_all_areas(icube):
    """Compute areas of all cubed sphere grid cells"""
    dA = np.zeros((icube, icube))
    gp = np.array([-PIQ + (PI / (2 * icube)) * k for k in range(icube + 1)])
    
    ang = np.zeros((icube + 1, icube + 1))
    for k1 in range(icube + 1):
        for k2 in range(icube + 1):
            ang[k1, k2] = math.acos(-math.sin(gp[k1]) * math.sin(gp[k2]))
    
    for k1 in range(icube):
        for k2 in range(icube):
            a1 = ang[k1, k2]
            a2 = PI - ang[k1+1, k2]
            a3 = PI - ang[k1, k2+1]
            a4 = ang[k1+1, k2+1]
            
            dA[k1, k2] = -2.0 * PI + a1 + a2 + a3 + a4
    
    # Debug check
    dbg1 = np.sum(dA)
    print(f"DAcube consistency: {dbg1 - 4.0*PI/6.0}")
    
    return dA

def write_cube(ncube, terr_cube, landfrac_cube, landm_coslat_cube, var30_cube, 
               raw_latlon_data_file, output_file):
    """Write cubed sphere data to NetCDF file"""
    grid_dims = 6 * ncube * ncube
    
    # Compute grid center coordinates
    grid_center_lat = np.zeros(grid_dims)
    grid_center_lon = np.zeros(grid_dims)
    
    da = PI / (2 * ncube)
    atm_add = 0
    
    for k in range(6):
        for j in range(ncube):
            ygno_ce = -PIQ + da * (j + 0.5)
            for i in range(ncube):
                xgno_ce = -PIQ + da * (i + 0.5)
                lon, lat = cubed_sphere_rll_from_abp(xgno_ce, ygno_ce, k+1)
                grid_center_lon[atm_add] = lon * RAD2DEG
                grid_center_lat[atm_add] = lat * RAD2DEG
                atm_add += 1
    
    # Create NetCDF file
    print(f"Create NetCDF file for output: {output_file}")
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ncfile:
        # Global attributes
        ncfile.title = 'equi-angular gnomonic cubed sphere grid'
        ncfile.raw_topo = raw_latlon_data_file
        ncfile.source_code = 'https://github.com/NCAR/Topo.git'
        ncfile.note = 'LANDM_COSLAT is only used in CAM4'
        ncfile.history = f'Written on date: {datetime.now().strftime("%Y%m%d")}'
        ncfile.author = 'Peter Hjort Lauritzen (NCAR)'
        
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
        lon_var.units = 'degrees_east'
        terr_var.units = 'm'
        landfrac_var.long_name = 'land ocean transition mask: ocean (0), continent (1), transition (0-1)'
        landm_coslat_var.long_name = 'smoothed land ocean transition mask'
        var30_var.units = 'm'
        var30_var.long_name = 'variance of elevation from high res lat-lon to ~3km cubed-sphere'
        
        # Write data
        grid_dims_var[:] = grid_dims
        lat_var[:] = grid_center_lat
        lon_var[:] = grid_center_lon
        terr_var[:] = terr_cube.flatten()
        landfrac_var[:] = landfrac_cube.flatten()
        landm_coslat_var[:] = landm_coslat_cube.flatten()
        var30_var[:] = var30_cube.flatten()

def main():
    """Main program"""
    config = read_config()
    ncube = config['ncube']
    raw_latlon_data_file = config['raw_latlon_data_file']
    output_file = config['output_file']
    
    print(f"Intermediate cubed-sphere resolution: {ncube}")
    
    # Read terrain data from NetCDF file
    print(f"Opening: {raw_latlon_data_file}")
    with nc.Dataset(raw_latlon_data_file, 'r') as ncfile:
        # Get dimensions
        im = len(ncfile.dimensions['lon'])
        jm = len(ncfile.dimensions['lat'])
        print(f"lon-lat dimensions: {im}, {jm}")
        
        # Read data
        landfrac = ncfile.variables['landfract'][:]
        terr = ncfile.variables['htopo'][:]
        lon = ncfile.variables['lon'][:]
        lat = ncfile.variables['lat'][:]
        
        print(f"min/max of 30sec land fraction: {np.min(landfrac)}, {np.max(landfrac)}")
    
    print("done reading data from netCDF file")
    
    # Compute volume for raw data
    print("compute volume for raw data")
    vol = 0.0
    dx = (lon[1] - lon[0])
    dx_rad = dx * DEG2RAD
    area_latlon = 0.0
    
    for j in range(jm):
        for i in range(im):
            darea_latlon = dx_rad * (math.sin(DEG2RAD*(-90.0 + dx*j)) - 
                                   math.sin(DEG2RAD*(-90.0 + dx*(j-1))))
            vol += float(terr[j, i]) * darea_latlon
            area_latlon += darea_latlon
    
    vol = vol / area_latlon
    print(f"consistency of lat-lon area: {area_latlon - 4.0*PI}")
    print(f"mean elevation (raw data): {vol}")
    
    # Read LANDM_COSLAT data
    print("read LANDM_COSLAT from file")
    with nc.Dataset('landm_coslat.nc', 'r') as ncfile:
        im_landm = len(ncfile.dimensions['lon'])
        jm_landm = len(ncfile.dimensions['lat'])
        print(f"lon-lat dimensions: {im_landm}, {jm_landm}")
        
        landm_coslat = ncfile.variables['LANDM_COSLAT'][:]
        lon_landm = ncfile.variables['lon'][:]
        lat_landm = ncfile.variables['lat'][:]
        
        print(f"min/max of landm_coslat: {np.min(landm_coslat)}, {np.max(landm_coslat)}")
    
    print("done reading in LANDM_COSLAT data from netCDF file")
    
    # Bin data to cubed-sphere grid
    da = PI / (2 * ncube)
    lon_rad = lon * DEG2RAD
    lat_rad = lat * DEG2RAD
    dlat = PI / jm
    
    # Initialize arrays
    weight = np.zeros((ncube, ncube, 6))
    terr_cube = np.zeros((ncube, ncube, 6))
    landfrac_cube = np.zeros((ncube, ncube, 6))
    landm_coslat_cube = np.zeros((ncube, ncube, 6))
    
    idx = np.zeros((im, jm), dtype=int)
    idy = np.zeros((im, jm), dtype=int)
    idp = np.zeros((im, jm), dtype=int)
    
    print("bin lat-lon data to cubed-sphere")
    
    for j in range(jm):
        for i in range(im):
            alpha, beta, ipanel = cubed_sphere_abp_from_rll(lon_rad[i], lat_rad[j])
            
            icube = math.ceil((alpha + PIQ) / da)
            jcube = math.ceil((beta + PIQ) / da)
            
            if icube < 1 or icube > ncube or jcube < 1 or jcube > ncube:
                print(f"fatal error: icube or jcube out of range: {icube}, {jcube}")
                sys.exit(1)
            
            # Convert to 0-based indexing
            icube -= 1
            jcube -= 1
            ipanel -= 1
            
            wt = math.sin(lat_rad[j] + 0.5*dlat) - math.sin(lat_rad[j] - 0.5*dlat)
            weight[icube, jcube, ipanel] += wt
            
            terr_cube[icube, jcube, ipanel] += wt * float(terr[j, i])
            landfrac_cube[icube, jcube, ipanel] += wt * float(landfrac[j, i])
            
            # Save indices for variance computation
            idx[i, j] = icube
            idy[i, j] = jcube
            idp[i, j] = ipanel
    
    # Normalize by weights and interpolate landm_coslat
    dx_landm = DEG2RAD * (lon_landm[1] - lon_landm[0])
    
    for k in range(6):
        for j in range(ncube):
            for i in range(ncube):
                if abs(weight[i, j, k]) < 1.0e-9:
                    print(f"Warning: no lat-lon grid point in cubed sphere cell {i}, {j}, {k}")
                else:
                    terr_cube[i, j, k] /= weight[i, j, k]
                    landfrac_cube[i, j, k] /= weight[i, j, k]
                
                # Linear interpolation for landm_coslat
                alpha = -PIQ + (i + 0.5) * da
                beta = -PIQ + (j + 0.5) * da
                lambda_coord, theta = cubed_sphere_rll_from_abp(alpha, beta, k+1)
                
                if theta > lat_landm[-1] * DEG2RAD - TINY:
                    landm_coslat_cube[i, j, k] = 0.0
                elif theta < lat_landm[0] * DEG2RAD + TINY:
                    landm_coslat_cube[i, j, k] = 1.0
                else:
                    # Simplified bilinear interpolation (would need proper implementation)
                    landm_coslat_cube[i, j, k] = 0.5  # Placeholder
    
    print(f"min/max value of terr_cube: {np.min(terr_cube)}, {np.max(terr_cube)}")
    print(f"min/max value of landm_coslat_cube: {np.min(landm_coslat_cube)}, {np.max(landm_coslat_cube)}")
    
    # Compute volume for cubed-sphere data
    print("compute volume for cubed-sphere binned data")
    darea_cube = equiangular_all_areas(ncube)
    
    vol_cube = 0.0
    for ipanel in range(6):
        for j in range(ncube):
            for i in range(ncube):
                vol_cube += terr_cube[i, j, ipanel] * darea_cube[i, j]
    
    vol_cube = vol_cube / (4.0 * PI)
    print(f"mean height (globally) of topography about sea-level (3km cube data): {vol_cube}, {(vol_cube-vol)/vol}")
    
    # Compute variance
    var30_cube = np.zeros((ncube, ncube, 6))
    
    for j in range(jm):
        for i in range(im):
            icube = idx[i, j]
            jcube = idy[i, j]
            ipanel = idp[i, j]
            wt = math.sin(lat_rad[j] + 0.5*dlat) - math.sin(lat_rad[j] - 0.5*dlat)
            var30_cube[icube, jcube, ipanel] += (wt * (terr_cube[icube, jcube, ipanel] - terr[j, i])**2) / weight[icube, jcube, ipanel]
    
    print(f"min/max value of var30_cube: {np.min(np.sqrt(var30_cube))}, {np.max(np.sqrt(var30_cube))}")
    
    # Write output file
    write_cube(ncube, terr_cube, landfrac_cube, landm_coslat_cube, var30_cube, 
               raw_latlon_data_file, output_file)
    
    print("done writing cubed sphere data")

if __name__ == "__main__":
    main()
