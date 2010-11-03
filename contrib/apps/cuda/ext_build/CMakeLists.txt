CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#------------------------------------------------------------------------------
# Field3D
#------------------------------------------------------------------------------

INCLUDE_DIRECTORIES( $ENV{FIELD3D_ROOT}/include )
LINK_DIRECTORIES( $ENV{FIELD3D_ROOT}/lib )
SET( pkgs_LIBRARIES ${pkgs_LIBRARIES} field3d )

#------------------------------------------------------------------------------
# HDF5
#------------------------------------------------------------------------------

FIND_PACKAGE( HDF5 )
INCLUDE_DIRECTORIES( ${HDF5_INCLUDE_DIR} )
LINK_DIRECTORIES ( ${HDF5_LIBRARY_DIRS} )
SET( pkgs_LIBRARIES ${pkgs_LIBRARIES} ${HDF5_LIBRARIES} )

#------------------------------------------------------------------------------
# Imath
#------------------------------------------------------------------------------

INCLUDE_DIRECTORIES ( $ENV{ILMBASE_ROOT}/include/OpenEXR )
INCLUDE_DIRECTORIES ( $ENV{ILMBASE_ROOT}/lib )
SET( pkgs_LIBRARIES ${pkgs_LIBRARIES} Imath Half )

#------------------------------------------------------------------------------
# Cuda
#------------------------------------------------------------------------------

FIND_PACKAGE(CUDA)

# Cuda SDK (for cutil)
INCLUDE_DIRECTORIES ( $ENV{NVSDKCUDA_ROOT}/C/common/inc )

# Thrust
INCLUDE_DIRECTORIES ( $ENV{THRUST_ROOT} )

# target architecture
IF( NOT DEFINED ENV{FIELD3D_CUDA_ARCH} )
  MESSAGE( FATAL_ERROR "Environment variable FIELD3D_CUDA_ARCH needs to be set (eg sm_11, sm_13, sm_20 etc)" )
ENDIF( NOT DEFINED ENV{FIELD3D_CUDA_ARCH} )
MESSAGE( STATUS "FIELD3D_CUDA_ARCH: $ENV{FIELD3D_CUDA_ARCH}" )
SET( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=$ENV{FIELD3D_CUDA_ARCH}" )

# verbose
#SET( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ptxas-options=-v" )

#indicate that we want Cuda code
ADD_DEFINITIONS( -DINCLUDE_FIELD3D_CUDA )

#------------------------------------------------------------------------------
# Boost
#------------------------------------------------------------------------------

FIND_PACKAGE( Boost COMPONENTS thread )
INCLUDE_DIRECTORIES ( ${Boost_INCLUDE_DIR} )
LINK_DIRECTORIES ( ${Boost_LIBRARY_DIRS} )
SET( pkgs_LIBRARIES ${pkgs_LIBRARIES} ${BOOST_LIBRARIES} )

#------------------------------------------------------------------------------
# OpenMP
#------------------------------------------------------------------------------

SET( CMAKE_CXX_FLAGS "-fopenmp")

#------------------------------------------------------------------------------

INCLUDE_DIRECTORIES ( .. )
INCLUDE_DIRECTORIES ( ../.. )
INCLUDE_DIRECTORIES ( ../../.. )

# ----------------------------------------------------------------------------

CUDA_ADD_EXECUTABLE ( field3d_cuda_simple
	../cuda_simple.cu
	../cuda_simple.cpp
)
TARGET_LINK_LIBRARIES ( field3d_cuda_simple ${pkgs_LIBRARIES} )
INSTALL ( TARGETS field3d_cuda_simple DESTINATION . )

# ----------------------------------------------------------------------------

CUDA_ADD_EXECUTABLE ( field3d_cuda_test
	../cuda_field_test.cu
	../cuda_field_test.cpp
)
TARGET_LINK_LIBRARIES ( field3d_cuda_test ${pkgs_LIBRARIES} )
INSTALL ( TARGETS field3d_cuda_test DESTINATION . )

# ----------------------------------------------------------------------------
