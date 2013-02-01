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

/*! \file SparseFieldIO.h
  \brief Contains the SparseFieldIO class.
  
  \todo Use boost::addressof instead of & operator
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_SparseFieldIO_H_
#define _INCLUDED_Field3D_SparseFieldIO_H_

//----------------------------------------------------------------------------//

#include <string>
#include <cmath>

#include <hdf5.h>

#include "SparseDataReader.h"
#include "SparseField.h"
#include "SparseFile.h"
#include "FieldIO.h"
#include "Field3DFile.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// SparseFieldIO
//----------------------------------------------------------------------------//

/*! \class SparseFieldIO
  \ingroup file_int
   Defines the IO for a SparseField object
 */

//----------------------------------------------------------------------------//

class SparseFieldIO : public FieldIO 
{

public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<SparseFieldIO> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef SparseFieldIO class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;

  const char *classType() const
  {
    return "SparseFieldIO";
  }
  
  // Constructors --------------------------------------------------------------

  //! Ctor
  SparseFieldIO() 
   : FieldIO()
  { }

  //! Dtor
  virtual ~SparseFieldIO() 
  { /* Empty */ }


  static FieldIO::Ptr create()
  { return Ptr(new SparseFieldIO); }

  // From FieldIO --------------------------------------------------------------

  //! Reads the field at the given location and tries to create a SparseField
  //! object from it.
  //! \returns Null if no object was read
  virtual FieldBase::Ptr read(hid_t layerGroup, const std::string &filename, 
                              const std::string &layerPath,
                              DataTypeEnum typeEnum);

  //! Writes the given field to disk. 
  //! \return true if successful, otherwise false
  virtual bool write(hid_t layerGroup, FieldBase::Ptr field);

  //! Returns the class name
  virtual std::string className() const
  { return "SparseField"; }

private:

  // Internal methods ----------------------------------------------------------

  //! This call writes all the attributes and sets up the data space.
  template <class Data_T>
  bool writeInternal(hid_t layerGroup, typename SparseField<Data_T>::Ptr field);

  //! Reads the data that is dependent on the data type on disk
  template <class Data_T>
  bool readData(hid_t location, 
                int numBlocks, 
                const std::string &filename, 
                const std::string &layerPath, 
                typename SparseField<Data_T>::Ptr result);

  // Strings -------------------------------------------------------------------

  static const int         k_versionNumber;
  static const std::string k_versionAttrName;
  static const std::string k_extentsStr;
  static const std::string k_dataWindowStr;
  static const std::string k_componentsStr;
  static const std::string k_blockOrderStr;
  static const std::string k_numBlocksStr;
  static const std::string k_blockResStr;
  static const std::string k_bitsPerComponentStr;
  static const std::string k_numOccupiedBlocksStr;
  static const std::string k_dataStr;
  
  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef FieldIO base;  
};

//----------------------------------------------------------------------------//
// Template methods
//----------------------------------------------------------------------------//

//! \todo Tune the chunk size of the gzip call
template <class Data_T>
bool SparseFieldIO::writeInternal(hid_t layerGroup, 
                                  typename SparseField<Data_T>::Ptr field)
{
  using namespace std;
  using namespace Exc;
  using namespace Hdf5Util;
  using namespace Sparse;

  Box3i ext(field->extents()), dw(field->dataWindow());

  int components = FieldTraits<Data_T>::dataDims();

  int valuesPerBlock = (1 << (field->m_blockOrder * 3)) * components;



  // Add extents attribute ---

  int extents[6] = 
    { ext.min.x, ext.min.y, ext.min.z, ext.max.x, ext.max.y, ext.max.z };

  if (!writeAttribute(layerGroup, k_extentsStr, 6, extents[0])) {
    Msg::print(Msg::SevWarning, "Error adding size attribute.");
    return false;
  }

  // Add data window attribute ---

  int dataWindow[6] = 
    { dw.min.x, dw.min.y, dw.min.z, dw.max.x, dw.max.y, dw.max.z };

  if (!writeAttribute(layerGroup, k_dataWindowStr, 6, dataWindow[0])) {
    Msg::print(Msg::SevWarning, "Error adding size attribute.");
    return false;
  }

  // Add components attribute ---

  if (!writeAttribute(layerGroup, k_componentsStr, 1, components)) {
    Msg::print(Msg::SevWarning, "Error adding components attribute.");
    return false;
  }

  // Add block order attribute ---

  int blockOrder = field->m_blockOrder;

  if (!writeAttribute(layerGroup, k_blockOrderStr, 1, blockOrder)) {
    Msg::print(Msg::SevWarning, "Error adding block order attribute.");
    return false;
  }

  // Add number of blocks attribute ---
  
  V3i &blockRes = field->m_blockRes;
  int numBlocks = blockRes.x * blockRes.y * blockRes.z;

  if (!writeAttribute(layerGroup, k_numBlocksStr, 1, numBlocks)) {
    Msg::print(Msg::SevWarning, "Error adding number of blocks attribute.");
    return false;
  }

  // Add block resolution in each dimension ---

  if (!writeAttribute(layerGroup, k_blockResStr, 3, blockRes.x)) {
    Msg::print(Msg::SevWarning, "Error adding block res attribute.");
    return false;
  }

  // Add the bits per component attribute ---

  int bits = DataTypeTraits<Data_T>::h5bits();
  if (!writeAttribute(layerGroup, k_bitsPerComponentStr, 1, bits)) {
    Msg::print(Msg::SevWarning, "Error adding bits per component attribute.");
    return false;    
  }

  // Write the block info data sets ---
  
  SparseBlock<Data_T> *blocks = field->m_blocks;

  // ... Write the isAllocated array
  {
    vector<char> isAllocated(numBlocks);
    for (int i = 0; i < numBlocks; ++i) {
      isAllocated[i] = static_cast<char>(blocks[i].isAllocated);
    }
    writeSimpleData<char>(layerGroup, "block_is_allocated_data", isAllocated);
  }

  // ... Write the emptyValue array
  {
    vector<Data_T> emptyValue(numBlocks);
    for (int i = 0; i < numBlocks; ++i) {
      emptyValue[i] = static_cast<Data_T>(blocks[i].emptyValue);
    }
    writeSimpleData<Data_T>(layerGroup, "block_empty_value_data", emptyValue);
  }

  // Count the number of occupied blocks ---
  int occupiedBlocks = 0;
  for (int i = 0; i < numBlocks; ++i) {
    if (blocks[i].isAllocated) {
      occupiedBlocks++;
  }
  }

  if (!writeAttribute(layerGroup, k_numOccupiedBlocksStr, 1, occupiedBlocks)) {
    throw WriteAttributeException("Couldn't add attribute " + 
                                k_numOccupiedBlocksStr);
  }
  
  if (occupiedBlocks > 0) {

    // Make the memory data space
    hsize_t memDims[1];
    memDims[0] = valuesPerBlock;
    H5ScopedScreate memDataSpace(H5S_SIMPLE);
    H5Sset_extent_simple(memDataSpace.id(), 1, memDims, NULL);

    // Make the file data space
    hsize_t fileDims[2];
    fileDims[0] = occupiedBlocks;
    fileDims[1] = valuesPerBlock;
    H5ScopedScreate fileDataSpace(H5S_SIMPLE);
    H5Sset_extent_simple(fileDataSpace.id(), 2, fileDims, NULL);

    // Set up gzip property list
    bool gzipAvailable = checkHdf5Gzip();
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunkSize[2];
    chunkSize[0] = 1;
    chunkSize[1] = valuesPerBlock;
    if (gzipAvailable) {
      herr_t status = H5Pset_deflate(dcpl, 9);
      if (status < 0) {
        return false;
      }
      status = H5Pset_chunk(dcpl, 2, chunkSize);
      if (status < 0) {
        return false;
      }    
    }

    // Add the data set
    H5ScopedDcreate dataSet(layerGroup, k_dataStr, 
                            DataTypeTraits<Data_T>::h5type(), 
                            fileDataSpace.id(), 
                            H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dataSet.id() < 0)
      throw CreateDataSetException("Couldn't create data set in "
                                   "SparseFieldIO::writeInternal");

    // For each allocated block ---

    int nextBlockIdx = 0;
    hsize_t offset[2];
    hsize_t count[2];
    herr_t status;

    for (int i = 0; i < numBlocks; ++i) {
      if (blocks[i].isAllocated) {
        offset[0] = nextBlockIdx;  // Index of next block
        offset[1] = 0;             // Index of first data in block. Always 0
        count[0] = 1;              // Number of columns to read. Always 1
        count[1] = valuesPerBlock; // Number of values in one column
        status = H5Sselect_hyperslab(fileDataSpace.id(), H5S_SELECT_SET, 
                                     offset, NULL, count, NULL);
        if (status < 0) {
          throw WriteHyperSlabException(
            "Couldn't select slab " + 
            boost::lexical_cast<std::string>(nextBlockIdx));
        }
        Data_T *data = field->m_blocks[i].data;
        status = H5Dwrite(dataSet.id(), DataTypeTraits<Data_T>::h5type(), 
                          memDataSpace.id(), 
                          fileDataSpace.id(), H5P_DEFAULT, data);
        if (status < 0) {
          throw WriteHyperSlabException(
            "Couldn't write slab " + 
            boost::lexical_cast<std::string>(nextBlockIdx));
        }
        // Increment nextBlockIdx
        nextBlockIdx++;
      }
    }

  } // if occupiedBlocks > 0

  return true; 

}

//----------------------------------------------------------------------------//

template <class Data_T>
bool SparseFieldIO::readData(hid_t location, 
                             int numBlocks, 
                             const std::string &filename, 
                             const std::string &layerPath, 
                             typename SparseField<Data_T>::Ptr result)
{
  using namespace std;
  using namespace Exc;
  using namespace Hdf5Util;
  using namespace Sparse;

  int occupiedBlocks;

  bool dynamicLoading = SparseFileManager::singleton().doLimitMemUse();

  int components = FieldTraits<Data_T>::dataDims();
  int valuesPerBlock = (1 << (result->m_blockOrder * 3)) * components;
  
  // Read the number of occupied blocks ---

  if (!readAttribute(location, k_numOccupiedBlocksStr, 1, occupiedBlocks)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_numOccupiedBlocksStr);

  // Set up the dynamic read info ---

  if (dynamicLoading) {
      // Set up the field reference
    result->addReference(filename, layerPath,
                         valuesPerBlock,
                         occupiedBlocks);
  }

  // Read the block info data sets ---

  SparseBlock<Data_T> *blocks = result->m_blocks;

  // ... Read the isAllocated array

  {
    vector<char> isAllocated(numBlocks);
    readSimpleData<char>(location, "block_is_allocated_data", isAllocated);
    for (int i = 0; i < numBlocks; ++i) {
      blocks[i].isAllocated = isAllocated[i];
      if (isAllocated[i]) {
        blocks[i].resize(valuesPerBlock);
      }
    }
  }

  // ... Read the emptyValue array ---

  {
    vector<Data_T> emptyValue(numBlocks);
    readSimpleData<Data_T>(location, "block_empty_value_data", emptyValue);
    for (int i = 0; i < numBlocks; ++i) {
      blocks[i].emptyValue = emptyValue[i];
    }
  }

  // Read the data ---

  if (occupiedBlocks > 0) {

    if (dynamicLoading) {

      result->setupReferenceBlocks();

    } else {
      
      size_t b = 0, bend = b + numBlocks;

      SparseDataReader<Data_T> reader(location, valuesPerBlock, occupiedBlocks);

      // We'll read at most 50meg at a time
      static const long maxMemPerPass = 50*1024*1024;

      for (int nextBlockIdx = 0;;) {

        long mem = 0;
        std::vector<Data_T*> memoryList;
        
        for (; b != bend && mem < maxMemPerPass; ++b) {
          if (blocks[b].isAllocated) {
            mem += sizeof(Data_T)*valuesPerBlock;
            memoryList.push_back(blocks[b].data);
          }
        }

        // all done.
        if (!memoryList.size()) {
          break;
        }

        reader.readBlockList(nextBlockIdx, memoryList);
        nextBlockIdx += memoryList.size();
      }                           

    }

  } // if occupiedBlocks > 0

  return true;
  
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif
