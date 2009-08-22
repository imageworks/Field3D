//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2009 Sony Pictures Imageworks
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the
 * distribution.  Neither the name of Sony Pictures Imageworks nor the
 * names of its contributors may be used to endorse or promote
 * products derived from this software without specific prior written
 * permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//----------------------------------------------------------------------------//

/*! \file Documentation.h
  \brief Only contains doxygen documentation. 
*/

//----------------------------------------------------------------------------//
// Modules
//----------------------------------------------------------------------------//

/*! \defgroup field Fields
  \brief These are the main parts of the library that a user would deal with.
*/

/*! \defgroup file File I/O
  \brief These are the main parts of the library that a user would deal with.
*/

/*! \defgroup internal Classes and functions used to implement this library
  \brief These classes are used to implement the library and are of interest
  to anyone maintaining or changing the library.
*/

/*! \defgroup exc Exceptions used in the library
  \ingroup internal
*/

/*! \defgroup hdf5 Hdf5 Utility Classes and Functions
  \ingroup internal
*/

/*! \defgroup file_int Field IO classes under-the-hood
  \ingroup internal
*/

/*! \defgroup field_int Field classes under-the-hood
  \ingroup internal
*/

/*! \defgroup template_util Templated utility classes
  \ingroup internal
*/

//----------------------------------------------------------------------------//
// Main Documentation page
//----------------------------------------------------------------------------//

/*! \mainpage Field3D
  
Field3D is an open source library for storing voxel data. It provides C++ 
classes that handle in-memory storage and a file format based on HDF5 that 
allows the C++ objects to be written to and read from disk.

\n

The majority of the documentation is available in the project Wiki at 
http://code.google.com/p/field3d/

*/ // \mainpage
