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

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake )

FIND_PACKAGE (Doxygen)
FIND_PACKAGE (HDF5)
IF ( CMAKE_HOST_WIN32 )
# f3dinfo relies on program_options but don't include it, since
# for some reason, unlike all the other boost components, a link is
# forced via a pragma.
FIND_PACKAGE (Boost COMPONENTS regex thread)
ELSE ()
FIND_PACKAGE (Boost COMPONENTS regex thread program_options system)
FIND_PACKAGE (MPI)
ENDIF ()

FIND_PACKAGE (ILMBase)

OPTION (INSTALL_DOCS "Automatically install documentation." ON)

# Duplicate the export directory to Field3D
FILE ( REMOVE_RECURSE ${CMAKE_HOME_DIRECTORY}/Field3D)
FILE ( COPY export/ DESTINATION ${CMAKE_HOME_DIRECTORY}/Field3D)

# includes
INCLUDE_DIRECTORIES ( . )
INCLUDE_DIRECTORIES ( src )
INCLUDE_DIRECTORIES ( export )
INCLUDE_DIRECTORIES ( ${ILMBASE_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES ( ${HDF5_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES ( ${Boost_INCLUDE_DIR} )

# link directories
LINK_DIRECTORIES ( ${Boost_LIBRARY_DIRS} )
LINK_DIRECTORIES ( ${HDF5_LIBRARY_DIRS} )
LINK_DIRECTORIES ( ${ILMBASE_LIBRARY_DIRS} )

IF ( CMAKE_HOST_UNIX )
  ADD_DEFINITIONS ( -fPIC -DREQUIRE_IOSTREAM -Wno-invalid-offsetof )
ENDIF ( )
IF ( CMAKE_HOST_WIN32 )
  ADD_DEFINITIONS ( 
    -DOPENEXR_DLL 
    -D_HDF5USEDLL_ 
    -DHDF5CPP_USEDLL 
    -D_HAS_ITERATOR_DEBUGGING=0 
    -D_CRT_SECURE_NO_WARNINGS=1
    )
ENDIF ( )

ADD_LIBRARY ( Field3D SHARED
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

SET ( Field3D_Libraries_Shared
  ${HDF5_LIBRARIES}
  )

IF ( CMAKE_HOST_UNIX )
  IF ( MPI_FOUND )
      LIST ( APPEND Field3D_Libraries_Shared
             ${MPI_LIBRARIES} )
  ENDIF ( MPI_FOUND )
  LIST ( APPEND Field3D_Libraries_Shared
    Iex Half IlmThread Imath
    pthread dl z )
  SET ( Field3D_DSO_Libraries ${Field3D_Libraries_Shared} )
  SET ( Field3D_BIN_Libraries Field3D ${Field3D_Libraries_Shared}
        ${Boost_LIBRARIES} )
ENDIF ( )
IF ( CMAKE_HOST_WIN32 )
  # Add OpenEXR and zlib release/debug
  FOREACH ( lib Iex Half IlmThread Imath zdll )
    LIST ( APPEND Field3D_Libraries_Shared
      optimized ${lib}
      debug ${lib}_d )
  ENDFOREACH()
  SET ( Field3D_DSO_Libraries ${Field3D_Libraries_Shared} )
  SET ( Field3D_BIN_Libraries Field3D ${Boost_LIBRARIES} )
ENDIF ()

TARGET_LINK_LIBRARIES ( Field3D ${Field3D_DSO_Libraries} )

# Parase version and soversion from export/ns.h

file(STRINGS export/ns.h FIELD3D_MAJOR_VER REGEX "^#define FIELD3D_MAJOR_VER")
file(STRINGS export/ns.h FIELD3D_MINOR_VER REGEX "^#define FIELD3D_MINOR_VER")
file(STRINGS export/ns.h FIELD3D_MICRO_VER REGEX "^#define FIELD3D_MICRO_VER")
string(REPLACE "#define FIELD3D_MAJOR_VER " "" FIELD3D_MAJOR_VER ${FIELD3D_MAJOR_VER})
string(REPLACE "#define FIELD3D_MINOR_VER " "" FIELD3D_MINOR_VER ${FIELD3D_MINOR_VER})
string(REPLACE "#define FIELD3D_MICRO_VER " "" FIELD3D_MICRO_VER ${FIELD3D_MICRO_VER})
SET ( FIELD3D_VERSION ${FIELD3D_MAJOR_VER}.${FIELD3D_MINOR_VER}.${FIELD3D_MICRO_VER} )
SET ( FIELD3D_SOVERSION ${FIELD3D_MAJOR_VER}.${FIELD3D_MINOR_VER} )

message(STATUS "Library soversion will be: ${FIELD3D_SOVERSION}")

SET_TARGET_PROPERTIES ( Field3D PROPERTIES VERSION ${FIELD3D_VERSION})
SET_TARGET_PROPERTIES ( Field3D PROPERTIES SOVERSION ${FIELD3D_SOVERSION})

IF ( CMAKE_HOST_WIN32 )
  SET_TARGET_PROPERTIES( Field3D PROPERTIES ENABLE_EXPORTS ON )
  SET_TARGET_PROPERTIES( Field3D PROPERTIES COMPILE_DEFINITIONS FIELD3D_EXPORT )
  SET_TARGET_PROPERTIES( Field3D PROPERTIES COMPILE_FLAGS -EHsc )
  SET_TARGET_PROPERTIES( Field3D PROPERTIES COMPILE_FLAGS -MD )
  SET_TARGET_PROPERTIES( Field3D PROPERTIES COMPILE_FLAGS -wd4251 )
ENDIF ( )



# field3d - unitTest
ADD_EXECUTABLE ( unitTest
  test/unit_tests/UnitTest.cpp
  )

TARGET_LINK_LIBRARIES ( unitTest ${Field3D_BIN_Libraries} )

IF ( CMAKE_HOST_WIN32 )
  SET_TARGET_PROPERTIES( unitTest PROPERTIES COMPILE_FLAGS -bigobj )
ENDIF ( )

# field3d - f3dinfo
ADD_EXECUTABLE ( f3dinfo
  apps/f3dinfo/main.cpp
  )

TARGET_LINK_LIBRARIES ( f3dinfo ${Field3D_BIN_Libraries} )

# field3d - sparse_field_io
ADD_EXECUTABLE ( sparse_field_io
  apps/sample_code/sparse_field_io/main.cpp
  )

TARGET_LINK_LIBRARIES ( sparse_field_io ${Field3D_BIN_Libraries} )

# field3d - read
ADD_EXECUTABLE ( read
  apps/sample_code/read/main.cpp
  )

TARGET_LINK_LIBRARIES ( read ${Field3D_BIN_Libraries} )

# field3d - mixed_types
ADD_EXECUTABLE ( mixed_types
  apps/sample_code/mixed_types/main.cpp
  )

TARGET_LINK_LIBRARIES ( mixed_types ${Field3D_BIN_Libraries} )

# field3d - create_and_write
ADD_EXECUTABLE ( create_and_write
  apps/sample_code/create_and_write/main.cpp
  )

TARGET_LINK_LIBRARIES ( create_and_write ${Field3D_BIN_Libraries} )

IF (DOXYGEN_FOUND)
  ADD_CUSTOM_TARGET ( doc
    ALL
    ${DOXYGEN_EXECUTABLE} Field3D.doxyfile
    WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}
    )
  IF (INSTALL_DOCS)
    INSTALL (DIRECTORY 
      ${CMAKE_HOME_DIRECTORY}/docs
      DESTINATION ${CMAKE_INSTALL_PREFIX}
    )
  ENDIF (INSTALL_DOCS)
ENDIF (DOXYGEN_FOUND)

INSTALL ( TARGETS
  Field3D
  DESTINATION lib${LIB_SUFFIX}
)

FILE(GLOB Field3d_Includes  "${CMAKE_CURRENT_SOURCE_DIR}/export/*.h")

INSTALL ( FILES 
  ${Field3d_Includes} 
  DESTINATION include/Field3D
)

INSTALL ( TARGETS f3dinfo 
  RUNTIME DESTINATION bin
)

