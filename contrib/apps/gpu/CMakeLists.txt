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
INCLUDE_DIRECTORIES ( $ENV{NVSDKCUDA_ROOT}/C/common/inc )
INCLUDE_DIRECTORIES ( $ENV{THRUST_ROOT} )

# double support
SET( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_13" )
# fermi build
#SET( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_20" )
# verbose
#SET( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ptxas-options=-v" )

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

INCLUDE_DIRECTORIES ( . )
INCLUDE_DIRECTORIES ( ../.. )

# ----------------------------------------------------------------------------

CUDA_ADD_EXECUTABLE ( gpu_simple
	gpu_simple.cu
	gpu_simple.cpp
)
TARGET_LINK_LIBRARIES ( gpu_simple ${pkgs_LIBRARIES} )
INSTALL ( TARGETS gpu_simple DESTINATION . )

# ----------------------------------------------------------------------------

CUDA_ADD_EXECUTABLE ( gpu_field_test
	gpu_field_test.cu
	gpu_field_test.cpp
)
TARGET_LINK_LIBRARIES ( gpu_field_test ${pkgs_LIBRARIES} )
INSTALL ( TARGETS gpu_field_test DESTINATION . )

# ----------------------------------------------------------------------------

#CUDA_ADD_EXECUTABLE ( cl_test
#	cl_test.cu
#	cl_test.cpp
#)
#TARGET_LINK_LIBRARIES ( cl_test ${pkgs_LIBRARIES} )
#INSTALL ( TARGETS cl_test DESTINATION . )

# ----------------------------------------------------------------------------
