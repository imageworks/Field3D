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

#include <zlib.h>

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
//! \note This class should not be accessed by multiple threads. Instead,
//! instantiate a separate one for each thread.
//! \ingroup file_int
template <class Data_T>
class OgSparseDataReader
{
public:

  // Constructors --------------------------------------------------------------

  //! Constructor. Requires knowledge of the Ogawa location where data
  //! is stored
  OgSparseDataReader(const OgIGroup &location, const size_t valuesPerBlock, 
                     const size_t occupiedBlocks, const bool isCompressed);

  // Main methods --------------------------------------------------------------

  //! Sets the reader's thread id
  void setThreadId(const size_t id);

  //! Reads a block, storing the data in result, which is assumed to contain
  //! enough room for m_valuesPerBlock entries.
  void readBlock(const size_t idx, Data_T *result);

private:

  // Data members --------------------------------------------------------------

  //! Og Dataset
  OgIDataset<Data_T> m_dataset;
  //! Compressed Og Dataset
  OgICDataset<Data_T> m_cDataset;
  //! Number of voxels per block
  const size_t m_numVoxels;
  //! Data name string
  const std::string k_dataStr;
  //! Whether the data is compressed
  const bool m_isCompressed;

  //! Thread ID. Used to alert Ogawa that different threads are accessing
  //! the API concurrently.
  size_t m_threadId;

  //! Cache for decompression
  std::vector<uint8_t> m_compressionCache;
};

//----------------------------------------------------------------------------//
// OgSparseDataReader implementations
//----------------------------------------------------------------------------//

template <class Data_T>
OgSparseDataReader<Data_T>::OgSparseDataReader(const OgIGroup &location, 
                                               const size_t numVoxels, 
                                               const size_t occupiedBlocks,
                                               const bool isCompressed) 
  : m_numVoxels(numVoxels), 
    k_dataStr("data"),
    m_isCompressed(isCompressed),
    m_threadId(0)
{
  using namespace Exc;

  if (isCompressed) {
    // Find the dataset
    m_cDataset = location.findCompressedDataset<Data_T>(k_dataStr);
    // Check validity
    if (!m_cDataset.isValid()) {
      throw ReadDataException("Couldn't open compressed data set: " + 
                              k_dataStr);
    }
    // Check element count
    if (m_cDataset.numDataElements() != occupiedBlocks) {
      throw ReadDataException("Block count mismatch in SparseDataReader");
    }
    // Check data type
    OgDataType typeOnDisk = location.compressedDatasetType(k_dataStr);
    if (typeOnDisk != OgawaTypeTraits<Data_T>::typeEnum()) {
      throw ReadDataException("Data type mismatch in SparseDataReader");
    }
    // Set the compresession cache size
    m_compressionCache.resize(compressBound(numVoxels * sizeof(Data_T)));
  } else {
    // Find the dataset
    m_dataset = location.findDataset<Data_T>(k_dataStr);
    // Check validity
    if (!m_dataset.isValid()) {
      throw ReadDataException("Couldn't open data set: " + k_dataStr);
    }
    // Check element count
    if (m_dataset.numDataElements() != occupiedBlocks) {
      throw ReadDataException("Block count mismatch in SparseDataReader");
    }
    // Check data type
    OgDataType typeOnDisk = location.datasetType(k_dataStr);
    if (typeOnDisk != OgawaTypeTraits<Data_T>::typeEnum()) {
      throw ReadDataException("Data type mismatch in SparseDataReader");
    }
  }

}

//----------------------------------------------------------------------------//

template <class Data_T>
void OgSparseDataReader<Data_T>::setThreadId(const size_t id)
{
  m_threadId = id;
}

//----------------------------------------------------------------------------//

template <class Data_T>
void OgSparseDataReader<Data_T>::readBlock(const size_t idx, Data_T *result)
{
  using namespace Exc;

  if (m_isCompressed) {

    // Length of compressed data
    const uint64_t length = m_cDataset.dataSize(idx, m_threadId);
    // Read data into compression cache
    m_cDataset.getData(idx, &m_compressionCache[0], m_threadId);
    // Target location
    uint8_t *ucmpData = reinterpret_cast<uint8_t *>(result);
    // Length of uncompressed data is stored here
    uLong ucmpLen = m_numVoxels * sizeof(Data_T);
    // Uncompress
    int status = uncompress(ucmpData, &ucmpLen, &m_compressionCache[0], 
                            length);
    if (status != Z_OK) {
      std::cout << "ERROR in uncompress: " << status
                << " " << ucmpLen << " " << length << std::endl;
      return;
    }

  } else {

    m_dataset.getData(idx, result, OGAWA_THREAD);

  }
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif
