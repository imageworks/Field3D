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

/*! \file SparseFile.h
  \brief Contains functions controlling the loading of sparse fields.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_SparseFile_H_
#define _INCLUDED_Field3D_SparseFile_H_

//----------------------------------------------------------------------------//

#include <vector>

#include <hdf5.h>

#include "Exception.h"
#include "Hdf5Util.h"
#include "SparseDataReader.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Forward declarations
//----------------------------------------------------------------------------//

namespace Sparse {

  template <typename Data_T>
  struct SparseBlock;

}

//----------------------------------------------------------------------------//

namespace SparseFile {

//----------------------------------------------------------------------------//
// Reference
//----------------------------------------------------------------------------//

/*! \class Reference
  Handles information about and block loading for a single SparseField 
  as referenced on disk
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class Reference 
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef std::vector<Sparse::SparseBlock<Data_T>*> BlockPtrs;

  // Public data members -------------------------------------------------------

  std::string filename;
  std::string layerPath;
  int valuesPerBlock;
  int occupiedBlocks;
 
  //! Index in file for each block
  std::vector<int> fileBlockIndices;
  //! Whether each block is loaded. We don't use bools since vector<bool> 
  //! is weird
  std::vector<int> blockLoaded;
  //! Pointers to each block. This is so we can go in and manipulate them
  //! as we please
  BlockPtrs blocks;

  // Ctors, dtor ---------------------------------------------------------------

  //! Constructor. Requires the filename and layer path of the field to be known
  Reference(const std::string filename, const std::string layerPath);
  ~Reference();

  // Main methods --------------------------------------------------------------

  //! Checks if the file used by this reference is open already
  bool fileIsOpen();
  //! Sets the number of blocks used by the SparseField we're supporting
  void setNumBlocks(int numBlocks);
  //! Opens the file. This is done just before the first request to loadBlock.
  //! This is delayed so that the original file open  has closed the file and
  //! doesn't cause any Hdf5 hiccups.
  void openFile();
  //! Loads the block with the given index into memory. We don't pass in 
  //! a reference to where the data should go since this is already know in the
  //! blocks data member.
  void loadBlock(int blockIdx);

private:

  //! Holds the Hdf5 handle to the file
  hid_t m_fileHandle;
  
  //! Hold the group containing the data open for the duration of the 
  //! Reference's existence.
  Hdf5Util::H5ScopedGopen m_layerGroup;

  //! Pointer to the reader object. NULL at construction time. Created in 
  //! openFile().
  SparseDataReader<Data_T> *m_reader;

};

//----------------------------------------------------------------------------//
// References
//----------------------------------------------------------------------------//

class FileReferences
{
public:

  // Main methods --------------------------------------------------------------

  //! Returns a reference to the index. This is specialized so that the
  //! correct data member is accessed.
  template <class Data_T>
  Reference<Data_T>& ref(int idx);

  //! Appends a reference to the collection. This is specialized so that the
  //! correct data member is accessed.
  template <class Data_T>
  int append(const Reference<Data_T>& ref);

private:

  // Data members --------------------------------------------------------------

  std::vector<Reference<half> > m_hRefs;
  std::vector<Reference<V3h> > m_vhRefs;
  std::vector<Reference<float> > m_fRefs;
  std::vector<Reference<V3f> > m_vfRefs;
  std::vector<Reference<double> > m_dRefs;
  std::vector<Reference<V3d> > m_vdRefs;

};

//----------------------------------------------------------------------------//

} // namespace SparseFile

//----------------------------------------------------------------------------//
// SparseFileManager
//----------------------------------------------------------------------------//

/*! \class SparseFileManager
  \ingroup file
  Handles specifics about reading sparse fields from disk. Its primary use
  is to control sparse fields read using memory limiting (dynamic loading).
*/

//----------------------------------------------------------------------------//

class SparseFileManager
{

public:

  // Main methods --------------------------------------------------------------

  //! Returns a reference to the singleton instance
  static SparseFileManager &singleton();

  //! Sets whether to limit memory usage and do dynamic loading for sparse 
  //! fields.
  void setLimitMemUse(bool enabled);

  //! Returns whether to limit memory usage and do dynamic loading for sparse 
  //! fields.
  bool doLimitMemUse() const;

  //! Sets the maximum memory usage by dynamically loaded sparse fields.
  //! \note This is not yet implemented.
  void setMaxMemUse(float maxMemUse);

  //! Returns the id of the next cache item. This is stored in the SparseField
  //! in order to reference its fields at a later time
  template <class Data_T>
  int getNextId(const std::string filename, const std::string layerPath);

  //! Returns a reference to the Reference object with the given index
  template <class Data_T>
  SparseFile::Reference<Data_T> &reference(int index);

  //! Called by SparseField when it's about to read from a block
  template <class Data_T>
  void activateBlock(int fileId, int blockIdx);

private:

  //! Private to prevent instantiation
  SparseFileManager();

  //! Pointer to singleton
  static SparseFileManager *ms_singleton;

  //! Max amount om memory to use in megabytes
  float m_maxMemUse;

  //! Whether to limit memory use of sparse fields from disk. Enables the
  //! cache and dynamic loading when true.
  bool m_limitMemUse;

  //! Vector containing information for each of the managed fields.
  //! The order matches the index stored in each SparseField::m_fileId
  SparseFile::FileReferences m_fileData;

};

//----------------------------------------------------------------------------//
// Reference implementations
//----------------------------------------------------------------------------//

namespace SparseFile {

//----------------------------------------------------------------------------//

template <class Data_T>
Reference<Data_T>::Reference(const std::string a_filename, 
                             const std::string a_layerPath)
  : filename(a_filename), layerPath(a_layerPath),
    valuesPerBlock(-1), occupiedBlocks(-1),
    m_fileHandle(-1), m_reader(NULL)    
{ 
  /* Empty */ 
}

//----------------------------------------------------------------------------//

template <class Data_T>
Reference<Data_T>::~Reference()
{
  if (m_reader)
    delete m_reader;
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool Reference<Data_T>::fileIsOpen()
{
  return m_fileHandle >= 0;
}

//----------------------------------------------------------------------------//

template <class Data_T>
void Reference<Data_T>::setNumBlocks(int numBlocks)
{
  fileBlockIndices.resize(numBlocks);
  blockLoaded.resize(numBlocks, 0);
  blocks.resize(numBlocks, 0);
}

//----------------------------------------------------------------------------//

template <class Data_T>
void Reference<Data_T>::openFile()
{
  using namespace Exc;
  using namespace Hdf5Util;

  m_fileHandle = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (m_fileHandle < 0)
    throw NoSuchFileException(filename);

  m_layerGroup.open(m_fileHandle, layerPath.c_str());
  if (m_layerGroup.id() < 0) {
    Log::print(Log::SevWarning, "In SparseFile::Reference::openFile: "
              "Couldn't find layer group " + layerPath + 
              " in .f3d file ");
    throw FileIntegrityException(filename);
  }

  m_reader = new SparseDataReader<Data_T>(m_layerGroup.id(), valuesPerBlock, 
                                          occupiedBlocks);
}

//----------------------------------------------------------------------------//

template <class Data_T>
void Reference<Data_T>::loadBlock(int blockIdx)
{
  // Allocate the block
  blocks[blockIdx]->data.resize(valuesPerBlock);
  assert(blocks[blockIdx]->data.size() > 0);
  // Read the data
  assert(m_reader);
  m_reader->readBlock(fileBlockIndices[blockIdx], blocks[blockIdx]->dataRef());
  // Mark block as loaded
  blockLoaded[blockIdx] = 1;
}

//----------------------------------------------------------------------------//

} // namespace SparseFile

//----------------------------------------------------------------------------//
// Specializations for FileReferences
//----------------------------------------------------------------------------//

namespace SparseFile {

//----------------------------------------------------------------------------//

template <>
inline Reference<half>& 
FileReferences::ref(int idx)
{
  return m_hRefs[idx];
}

//----------------------------------------------------------------------------//

template <>
inline Reference<V3h>& 
FileReferences::ref(int idx)
{
  return m_vhRefs[idx];
}

//----------------------------------------------------------------------------//

template <>
inline Reference<float>& 
FileReferences::ref(int idx)
{
  return m_fRefs[idx];
}

//----------------------------------------------------------------------------//

template <>
inline Reference<V3f>& 
FileReferences::ref(int idx)
{
  return m_vfRefs[idx];
}

//----------------------------------------------------------------------------//

template <>
inline Reference<double>& 
FileReferences::ref(int idx)
{
  return m_dRefs[idx];
}

//----------------------------------------------------------------------------//

template <>
inline Reference<V3d>& 
FileReferences::ref(int idx)
{
  return m_vdRefs[idx];
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::append(const Reference<half>& ref)
{
  m_hRefs.push_back(ref);
  return m_hRefs.size() - 1;
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::append(const Reference<V3h>& ref)
{
  m_vhRefs.push_back(ref);
  return m_vhRefs.size() - 1;
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::append(const Reference<float>& ref)
{
  m_fRefs.push_back(ref);
  return m_fRefs.size() - 1;
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::append(const Reference<V3f>& ref)
{
  m_vfRefs.push_back(ref);
  return m_vfRefs.size() - 1;
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::append(const Reference<double>& ref)
{
  m_dRefs.push_back(ref);
  return m_dRefs.size() - 1;
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::append(const Reference<V3d>& ref)
{
  m_vdRefs.push_back(ref);
  return m_vdRefs.size() - 1;
}

//----------------------------------------------------------------------------//
// Implementations for FileReferences
//----------------------------------------------------------------------------//

template <class Data_T>
Reference<Data_T>& FileReferences::ref(int idx)
{
  assert(false && "Do not use memory limiting on sparse fields that aren't "
         "simple scalars or vectors!");
  Log::print(Log::SevWarning, 
             "FileReferences::ref(): Do not use memory limiting on sparse "
             "fields that aren't simple scalars or vectors!");
  static Reference<Data_T> dummy("", "");
  return dummy;
}

//----------------------------------------------------------------------------//

template <class Data_T>
int FileReferences::append(const Reference<Data_T>& ref)
{
  assert(false && "Do not use memory limiting on sparse fields that aren't "
         "simple scalars or vectors!");
  Log::print(Log::SevWarning,
             "FileReferences::append(): Do not use memory limiting on sparse "
             "fields that aren't simple scalars or vectors!");
  return -1;    
}

//----------------------------------------------------------------------------//

} // namespace SparseFile

//----------------------------------------------------------------------------//
// SparseFileManager implementations
//----------------------------------------------------------------------------//

template <class Data_T>
int 
SparseFileManager::getNextId(const std::string filename, 
                             const std::string layerPath)
{
  using namespace SparseFile;

  int id = m_fileData.append(Reference<Data_T>(filename, layerPath));
  return id;
}

//----------------------------------------------------------------------------//

template <class Data_T>
SparseFile::Reference<Data_T> &
SparseFileManager::reference(int index)
{ 
  return m_fileData.ref<Data_T>(index); 
}

//----------------------------------------------------------------------------//

template <class Data_T>
void 
SparseFileManager::activateBlock(int fileId, int blockIdx)
{
  SparseFile::Reference<Data_T> &reference = m_fileData.ref<Data_T>(fileId);

  if (reference.fileBlockIndices[blockIdx] >= 0 &&
      !reference.blockLoaded[blockIdx]) {
    if (!reference.fileIsOpen()) {
      reference.openFile();
    }
    reference.loadBlock(blockIdx);
  }
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif
