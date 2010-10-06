# Copyright (c) 2009 Sony Pictures Imageworks Inc. et al.
# 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the
# distribution. Neither the name of Sony Pictures Imageworks nor the
# names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written
# permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

# Author : Nicholas Yue yue.nicholas@gmail.com

CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

PROJECT ( field3d )

FIND_PACKAGE (Doxygen)
FIND_PACKAGE (HDF5)
FIND_PACKAGE( Boost COMPONENTS thread )

# Duplicate the export directory to Field3D
FILE ( REMOVE_RECURSE ${CMAKE_HOME_DIRECTORY}/Field3D)
FILE ( COPY export/ DESTINATION ${CMAKE_HOME_DIRECTORY}/Field3D)

# includes
INCLUDE_DIRECTORIES ( . )
INCLUDE_DIRECTORIES ( src )
INCLUDE_DIRECTORIES ( export )
INCLUDE_DIRECTORIES ( $ENV{ILMBASE_ROOT}/include )
INCLUDE_DIRECTORIES ( $ENV{ILMBASE_ROOT}/include/OpenEXR )
INCLUDE_DIRECTORIES ( ${HDF5_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES ( ${Boost_INCLUDE_DIR} )

# link directories
LINK_DIRECTORIES ( ${Boost_LIBRARY_DIRS} )
LINK_DIRECTORIES ( ${HDF5_LIBRARY_DIRS} )
LINK_DIRECTORIES ( $ENV{ILMBASE_ROOT}/lib )

IF ( LINUX )
  ADD_DEFINITIONS ( -fPIC -m64 -DLINUX -DREQUIRE_IOSTREAM -Wno-invalid-offsetof )
ENDIF ( LINUX )

ADD_LIBRARY ( field3d 
  src/ClassFactory.cpp
  src/DenseFieldIO.cpp
  src/Field3DFile.cpp
  src/Field.cpp
  src/FieldInterp.cpp
  src/FieldMapping.cpp
  src/FieldMappingIO.cpp
  src/Hdf5Util.cpp
  src/InitIO.cpp
  src/Log.cpp
  src/MACFieldIO.cpp
  src/PluginLoader.cpp
  src/SparseFieldIO.cpp
  src/SparseFile.cpp
)

# field3d - unitTest
ADD_EXECUTABLE ( unitTest
  test/unit_tests/UnitTest.cpp
  )

TARGET_LINK_LIBRARIES ( unitTest
  field3d
  ${HDF5_LIBRARIES}
  z
  Iex
  Half
  IlmThread
  Imath
  dl
  ${BOOST_LIBRARIES}
  )

# field3d - access_speed
ADD_EXECUTABLE ( access_speed
  apps/access_speed/main.cpp
  )

TARGET_LINK_LIBRARIES ( access_speed
  field3d
  ${HDF5_LIBRARIES}
  z
  Iex
  Half
  IlmThread
  Imath
  dl
  ${BOOST_LIBRARIES}
  )

# field3d - sparse_field_io
ADD_EXECUTABLE ( sparse_field_io
  apps/sparse_field_io/main.cpp
  )

TARGET_LINK_LIBRARIES ( sparse_field_io
  field3d
  ${HDF5_LIBRARIES}
  z
  Iex
  Half
  IlmThread
  Imath
  dl
  ${BOOST_LIBRARIES}
  )

# field3d - read
ADD_EXECUTABLE ( read
  apps/sample_code/read/main.cpp
  )

TARGET_LINK_LIBRARIES ( read
  field3d
  ${HDF5_LIBRARIES}
  z
  Iex
  Half
  IlmThread
  Imath
  dl
  ${BOOST_LIBRARIES}
  )

# field3d - mixed_types
ADD_EXECUTABLE ( mixed_types
  apps/sample_code/mixed_types/main.cpp
  )

TARGET_LINK_LIBRARIES ( mixed_types
  field3d
  ${HDF5_LIBRARIES}
  z
  Iex
  Half
  IlmThread
  Imath
  dl
  ${BOOST_LIBRARIES}
  )

# field3d - create_and_write
ADD_EXECUTABLE ( create_and_write
  apps/sample_code/create_and_write/main.cpp
  )

TARGET_LINK_LIBRARIES ( create_and_write
  field3d
  ${HDF5_LIBRARIES}
  z
  Iex
  Half
  IlmThread
  Imath
  dl
  ${BOOST_LIBRARIES}
  )

IF (DOXYGEN_FOUND)
  ADD_CUSTOM_TARGET ( doc
    ALL
    ${DOXYGEN_EXECUTABLE} Field3D.doxyfile
    WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}
    )
ENDIF (DOXYGEN_FOUND)

INSTALL ( TARGETS
  field3d
  DESTINATION lib
)

INSTALL ( FILES
  export/StdMathLib.h
  export/MACFieldIO.h
  export/Types.h
  export/FieldIO.h
  export/PluginLoader.h
  export/EmptyField.h
  export/Exception.h
  export/SpiMathLib.h
  export/MACField.h
  export/DenseField.h
  export/DataTypeConversion.h
  export/Hdf5Util.h
  export/SparseField.h
  export/FieldMapping.h
  export/Log.h
  export/Field3DFile.h
  export/ClassFactory.h
  export/RefCount.h
  export/Field.h
  export/FieldMappingIO.h
  export/InitIO.h
  export/ns.h
  export/SparseFile.h
  export/ProceduralField.h
  export/FieldInterp.h
  export/Documentation.h
  export/SparseDataReader.h
  export/SparseFieldIO.h
  export/MACFieldUtil.h
  export/DenseFieldIO.h
  DESTINATION include/Field3D
)

INSTALL (DIRECTORY ${CMAKE_HOME_DIRECTORY}/docs
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)
