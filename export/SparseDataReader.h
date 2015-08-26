//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2009 Sony Pictures Imageworks Inc
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
#include <string.h> // for memcpy

#include "Hdf5Util.h"
#include "Log.h"

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

  //! Reads a series of blocks, storing each block of data in memoryList, 
  //! which is assumed to contain enough room for m_valuesPerBlock entries.
  void readBlockList(int idx, const std::vector<Data_T*>& memoryList);

private:

  // Data members --------------------------------------------------------------

  hid_t m_location;

  int m_valuesPerBlock;
  int m_occupiedBlocks;

  const std::string k_dataStr;
};

//----------------------------------------------------------------------------//
// SparseDataReader implementations
//----------------------------------------------------------------------------//

template <class Data_T>
SparseDataReader<Data_T>::SparseDataReader(hid_t location, int valuesPerBlock, 
                                           int occupiedBlocks) 
  : m_location(location), 
    m_valuesPerBlock(valuesPerBlock), 
    m_occupiedBlocks(occupiedBlocks), 
    k_dataStr("data")
{

}

//----------------------------------------------------------------------------//

template <class Data_T>
void SparseDataReader<Data_T>::readBlock(int idx, Data_T &result)
{
  using namespace Hdf5Util;
  using namespace Exc;

  GlobalLock lock(g_hdf5Mutex);

  Hdf5Util::H5ScopedDopen      dataSet;
  Hdf5Util::H5ScopedDget_space fileDataSpace;
  Hdf5Util::H5ScopedDget_type  dataType;
  Hdf5Util::H5ScopedScreate    memDataSpace;

  hsize_t dims[2];
  hsize_t memDims[1];

  // Open the data set
  dataSet.open(m_location, k_dataStr, H5P_DEFAULT);
  if (dataSet.id() < 0) 
    throw OpenDataSetException("Couldn't open data set: " + k_dataStr);
    
  // Get the space and type
  fileDataSpace.open(dataSet.id());
  dataType.open(dataSet.id());
  if (fileDataSpace.id() < 0) 
    throw GetDataSpaceException("Couldn't get data space");
  if (dataType.id() < 0)
    throw GetDataTypeException("Couldn't get data type");

  // Make the memory data space
  memDims[0] = m_valuesPerBlock;
  memDataSpace.create(H5S_SIMPLE);
  H5Sset_extent_simple(memDataSpace.id(), 1, memDims, NULL);

  // Get the dimensions and check they match
  H5Sget_simple_extent_dims(fileDataSpace.id(), dims, NULL);
  if (dims[1] != static_cast<hsize_t>(m_valuesPerBlock)) {
    throw FileIntegrityException("Block length mismatch in "
                                 "SparseDataReader");
  }
  if (dims[0] != static_cast<hsize_t>(m_occupiedBlocks)) 
    throw FileIntegrityException("Block count mismatch in "
                                 "SparseDataReader");

  hsize_t offset[2];
  hsize_t count[2];
  herr_t status;
    
  offset[0] = idx;             // Index of block
  offset[1] = 0;               // Index of first data in block. Always 0
  count[0] = 1;                // Number of columns to read. Always 1
  count[1] = m_valuesPerBlock; // Number of values in one column

  status = H5Sselect_hyperslab(fileDataSpace.id(), H5S_SELECT_SET, 
                               offset, NULL, count, NULL);
    
  if (status < 0) {
    throw ReadHyperSlabException("Couldn't select slab in readBlock(): " + 
                                 boost::lexical_cast<std::string>(idx));
  }

  status = H5Dread(dataSet.id(), DataTypeTraits<Data_T>::h5type(), 
                   memDataSpace.id(), fileDataSpace.id(), 
                   H5P_DEFAULT, &result);
}

//----------------------------------------------------------------------------//

template <class Data_T>
void SparseDataReader<Data_T>::readBlockList
(int idxLo, const std::vector<Data_T*>& memoryList)
{
  using namespace Hdf5Util;
  using namespace Exc;

  GlobalLock lock(g_hdf5Mutex);

  Hdf5Util::H5ScopedDopen dataSet;
  Hdf5Util::H5ScopedDget_space fileDataSpace;
  Hdf5Util::H5ScopedDget_type dataType;
  Hdf5Util::H5ScopedScreate memDataSpace;

  hsize_t dims[2];
  hsize_t memDims[1];

  // Open the data set
  dataSet.open(m_location, k_dataStr, H5P_DEFAULT);
  if (dataSet.id() < 0) 
    throw OpenDataSetException("Couldn't open data set: " + k_dataStr);
    
  // Get the space and type
  fileDataSpace.open(dataSet.id());
  dataType.open(dataSet.id());
  if (fileDataSpace.id() < 0) 
    throw GetDataSpaceException("Couldn't get data space");
  if (dataType.id() < 0)
    throw GetDataTypeException("Couldn't get data type");

  // Make the memory data space
  memDims[0] = m_valuesPerBlock;
  memDataSpace.create(H5S_SIMPLE);
  H5Sset_extent_simple(memDataSpace.id(), 1, memDims, NULL);

  // Get the dimensions and check they match
  H5Sget_simple_extent_dims(fileDataSpace.id(), dims, NULL);
  if (dims[1] != static_cast<hsize_t>(m_valuesPerBlock)) {
    throw FileIntegrityException("Block length mismatch in "
                                 "SparseDataReader");
  }
  if (dims[0] != static_cast<hsize_t>(m_occupiedBlocks)) 
    throw FileIntegrityException("Block count mismatch in "
                                 "SparseDataReader");

  hsize_t offset[2];
  hsize_t count[2];
  herr_t status;

  offset[0] = idxLo;            // Index of block
  offset[1] = 0;                // Index of first data in block. Always 0
  count[0] = memoryList.size(); // Number of columns to read.
  count[1] = m_valuesPerBlock;  // Number of values in one column
  
  status = H5Sselect_hyperslab(fileDataSpace.id(), H5S_SELECT_SET, 
                               offset, NULL, count, NULL);
  if (status < 0) {
    throw ReadHyperSlabException("Couldn't select slab in readBlockList():" + 
                                 boost::lexical_cast<std::string>(idxLo));
  }

  // Make the memory data space ---
 
  Hdf5Util::H5ScopedScreate localMemDataSpace;
  hsize_t fileDims[2];  
  fileDims[0] = memoryList.size();
  fileDims[1] = m_valuesPerBlock;
  localMemDataSpace.create(H5S_SIMPLE);
  H5Sset_extent_simple(localMemDataSpace.id(), 2, fileDims, NULL);

  // Setup the temporary memory region ---

  int bytesPerValue = 0;
  {
    hid_t t = DataTypeTraits<Data_T>::h5type();
    if (t == H5T_NATIVE_CHAR)
      bytesPerValue = 1;
    else if (t == H5T_NATIVE_SHORT)
      bytesPerValue = 2;
    else if (t == H5T_NATIVE_FLOAT)
      bytesPerValue = 4;
    else if (t == H5T_NATIVE_DOUBLE)
      bytesPerValue = 8;
  }

  int dim = sizeof(Data_T) / bytesPerValue;
  std::vector<Data_T> bigblock(memoryList.size() * m_valuesPerBlock/dim);

  status = H5Dread(dataSet.id(), 
                   DataTypeTraits<Data_T>::h5type(), 
                   localMemDataSpace.id(),
                   fileDataSpace.id(), 
                   H5P_DEFAULT, &bigblock[0]);

  if (status < 0) {
    throw Hdf5DataReadException("Couldn't read slab " + 
                                boost::lexical_cast<std::string>(idxLo));
  }

  // Distribute block data into memory slots ---
  for (size_t i = 0; i < memoryList.size(); ++i) {
    memcpy(memoryList[i], 
           &bigblock[i * m_valuesPerBlock / dim],
           bytesPerValue * m_valuesPerBlock);
  }
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif
