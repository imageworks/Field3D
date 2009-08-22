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

/*! \file SparseDataReader.h
  \brief Contains functions controlling the loading of sparse fields.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_SparseDataReader_H_
#define _INCLUDED_Field3D_SparseDataReader_H_

//----------------------------------------------------------------------------//

#include <hdf5.h>

#include "Hdf5Util.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// SparseDataReader
//----------------------------------------------------------------------------//

//! This class gets used by SparseFieldIO and SparseFileManager to read
//! the block data. On creation it will open the data set and not close
//! it until the object is destroyed
//! \ingroup file_int
template <class Data_T>
class SparseDataReader
{
public:

  // Constructors --------------------------------------------------------------

  //! Constructor. Requires knowledge of the Hdf5 location where data
  //! is stored
  SparseDataReader(hid_t location, int valuesPerBlock, int occupiedBlocks);

  // Main methods --------------------------------------------------------------

  //! Reads a block, storing the data in result, which is assumed to contain
  //! enough room for m_valuesPerBlock entries.
  void readBlock(int idx, Data_T &result);

private:

  // Data members --------------------------------------------------------------

  Hdf5Util::H5ScopedDopen m_dataSet;
  Hdf5Util::H5ScopedDget_space m_fileDataSpace;
  Hdf5Util::H5ScopedDget_type m_dataType;
  Hdf5Util::H5ScopedScreate m_memDataSpace;
  
  int m_valuesPerBlock;

  const std::string k_dataStr;
};

//----------------------------------------------------------------------------//
// SparseDataReader implementations
//----------------------------------------------------------------------------//

template <class Data_T>
SparseDataReader<Data_T>::SparseDataReader(hid_t location, int valuesPerBlock, 
                                           int occupiedBlocks) 
  : m_valuesPerBlock(valuesPerBlock), 
    k_dataStr("data")
{
  using namespace Hdf5Util;
  using namespace Exc;

  hsize_t dims[2];
  hsize_t memDims[1];

  // Open the data set
  m_dataSet.open(location, k_dataStr, H5P_DEFAULT);
  if (m_dataSet.id() < 0) 
    throw OpenDataSetException("Couldn't open data set: " + k_dataStr);
    
  // Get the space and type
  m_fileDataSpace.open(m_dataSet.id());
  m_dataType.open(m_dataSet.id());
  if (m_fileDataSpace.id() < 0) 
    throw GetDataSpaceException("Couldn't get data space");
  if (m_dataType.id() < 0)
    throw GetDataTypeException("Couldn't get data type");

  // Make the memory data space
  memDims[0] = m_valuesPerBlock;
  m_memDataSpace.create(H5S_SIMPLE);
  H5Sset_extent_simple(m_memDataSpace.id(), 1, memDims, NULL);

  // Get the dimensions and check they match
  H5Sget_simple_extent_dims(m_fileDataSpace.id(), dims, NULL);
  if (dims[1] != static_cast<hsize_t>(m_valuesPerBlock)) {
    throw FileIntegrityException("Block length mismatch in "
                                 "SparseDataReader");
  }
  if (dims[0] != static_cast<hsize_t>(occupiedBlocks)) 
    throw FileIntegrityException("Block count mismatch in "
                                 "SparseDataReader");
}

//----------------------------------------------------------------------------//

template <class Data_T>
void SparseDataReader<Data_T>::readBlock(int idx, Data_T &result)
{
  using namespace Hdf5Util;
  using namespace Exc;

  hsize_t offset[2];
  hsize_t count[2];
  herr_t status;
    
  offset[0] = idx;             // Index of block
  offset[1] = 0;               // Index of first data in block. Always 0
  count[0] = 1;                // Number of columns to read. Always 1
  count[1] = m_valuesPerBlock; // Number of values in one column
  status = H5Sselect_hyperslab(m_fileDataSpace.id(), H5S_SELECT_SET, 
                               offset, NULL, count, NULL);
  if (status < 0) {
    throw ReadHyperSlabException("Couldn't select slab " + str(idx));
  }

  status = H5Dread(m_dataSet.id(), TypeToH5Type<Data_T>::type(), 
                   m_memDataSpace.id(), m_fileDataSpace.id(), 
                   H5P_DEFAULT, &result);
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif
