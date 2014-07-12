//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2014 Sony Pictures Imageworks Inc
 *                    Pixar Animation Studios
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

/*! \file OgSparseDataReader.h
  \brief Contains functions controlling the loading of sparse fields.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_OgSparseDataReader_H_
#define _INCLUDED_Field3D_OgSparseDataReader_H_

//----------------------------------------------------------------------------//

#include <hdf5.h>
#include <string.h> // for memcpy

#include "OgIO.h"
#include "Hdf5Util.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// OgSparseDataReader
//----------------------------------------------------------------------------//

//! This class gets used by SparseFieldIO and SparseFileManager to read
//! the block data. On creation it will open the data set and not close
//! it until the object is destroyed
//! \ingroup file_int
template <class Data_T>
class OgSparseDataReader
{
public:

  // Constructors --------------------------------------------------------------

  //! Constructor. Requires knowledge of the Ogawa location where data
  //! is stored
  OgSparseDataReader(const OgIGroup &location, const size_t valuesPerBlock, 
                     const size_t occupiedBlocks);

  // Main methods --------------------------------------------------------------

  //! Reads a block, storing the data in result, which is assumed to contain
  //! enough room for m_valuesPerBlock entries.
  void readBlock(const size_t idx, Data_T *result);

private:

  // Data members --------------------------------------------------------------

  OgIDataset<Data_T> m_dataset;
  
  const size_t m_numVoxels;

  const std::string k_dataStr;
};

//----------------------------------------------------------------------------//
// OgSparseDataReader implementations
//----------------------------------------------------------------------------//

template <class Data_T>
OgSparseDataReader<Data_T>::OgSparseDataReader(const OgIGroup &location, 
                                               const size_t numVoxels, 
                                               const size_t occupiedBlocks) 
  : m_numVoxels(numVoxels), 
    k_dataStr("data")
{
  using namespace Exc;

  m_dataset = location.findDataset<Data_T>(k_dataStr);

  if (!m_dataset.isValid()) {
    throw ReadDataException("Couldn't open data set: " + k_dataStr);
  }

  OgDataType typeOnDisk = location.datasetType(k_dataStr);
  if (typeOnDisk != OgawaTypeTraits<Data_T>::typeEnum()) {
    throw ReadDataException("Data type mismatch in SparseDataReader");
  }

  if (m_dataset.numDataElements() != occupiedBlocks) {
    throw ReadDataException("Block count mismatch in SparseDataReader");
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
void OgSparseDataReader<Data_T>::readBlock(const size_t idx, Data_T *result)
{
  using namespace Exc;

  m_dataset.getData(idx, result, OGAWA_THREAD);
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif
