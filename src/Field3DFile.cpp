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

/*! \file Field3DFile.cpp
  \brief Contains implementations of Field3DFile-related member functions
  \ingroup field
*/

//----------------------------------------------------------------------------//

#include <sys/stat.h>
#include <unistd.h>

#include <hdf5.h>
#include <H5Epublic.h>

#include <boost/tokenizer.hpp>
#include <boost/utility.hpp>

#include "Field3DFile.h"
#include "Field.h"
#include "ClassFactory.h"


//----------------------------------------------------------------------------//

using namespace std;

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Field3D namespaces
//----------------------------------------------------------------------------//

using namespace Exc;
using namespace Hdf5Util;
using namespace File;

//----------------------------------------------------------------------------//
// Local namespace
//----------------------------------------------------------------------------//

namespace {
  
  // Strings used only in this file --------------------------------------------

  const std::string k_mappingStr("mapping");
  const std::string k_versionAttrName("version_number");
  const std::string k_classNameAttrName("class_name");
  const std::string k_mappingTypeAttrName("mapping_type");

  //! This version is stored in every file to determine which library version
  //! produced it.

  int k_currentFileVersion[3] =
    { FIELD3D_MAJOR_VER, FIELD3D_MINOR_VER, FIELD3D_MICRO_VER };
  int k_minFileVersion[2] = { 0, 0 };

  // Function objects used only in this file -----------------------------------

  std::vector<std::string> makeUnique(std::vector<std::string> vec)
  {
    std::vector<string> ret;
    std::sort(vec.begin(), vec.end());
    std::vector<std::string>::iterator newEnd = 
      std::unique(vec.begin(), vec.end());
    ret.resize(std::distance(vec.begin(), newEnd));
    std::copy(vec.begin(), newEnd, ret.begin()); 
    return ret;
  }

//----------------------------------------------------------------------------//

  //! Functor used with for_each to print a container
  template <class T>
  class print : std::unary_function<T, void>
  {
  public:
    print(int indentAmt)
      : indent(indentAmt)
    { }
    void operator()(const T& x) const
    {
      for (int i = 0; i < indent; i++)
        std::cout << " ";
      std::cout << x << std::endl;
    }
    int indent;
  };

//----------------------------------------------------------------------------//

  /*! \brief checks to see if a file/directory exists or not
    \param[in] filename the file/directory to check
    \retval true if it exists
    \retval false if it does not exist
   */
  bool fileExists(const std::string &filename)
  {
    struct stat statbuf;
    return (stat(filename.c_str(), &statbuf) != -1);
  }

  /*! \brief wrapper around fileExists. Throws instead if the file
    does not exist.
    \throw NoSuchFileException if the file or directory does not exist
    \param[in] filename the file/directory to check
   */
  void checkFile(const std::string &filename)
  {
    if (!fileExists(filename))
    {
      throw NoSuchFileException(filename);
    }
  }

//----------------------------------------------------------------------------//

  bool isSupportedFileVersion(const int fileVersion[3],
                              const int minVersion[2])
  {
    stringstream currentVersionStr;
    currentVersionStr << k_currentFileVersion[0] << "."
                      << k_currentFileVersion[1] << "."
                      << k_currentFileVersion[2];
    stringstream fileVersionStr;
    fileVersionStr << fileVersion[0] << "."
                   << fileVersion[1] << "."
                   << fileVersion[2];
    stringstream minVersionStr;
    minVersionStr << minVersion[0] << "."
                  << minVersion[1];

    if (fileVersion[0] > k_currentFileVersion[0] ||
        (fileVersion[0] == k_currentFileVersion[0] && 
         fileVersion[1] > k_currentFileVersion[1])) {
      Msg::print(Msg::SevWarning, "File version " + fileVersionStr.str() +
                 " is higher than the current version " +
                 currentVersionStr.str());
      return true;
    }

    if (fileVersion[0] < minVersion[0] ||
        (fileVersion[0] == minVersion[0] &&
         fileVersion[1] < minVersion[1])) {
      Msg::print(Msg::SevWarning, "File version " + fileVersionStr.str() +
                 " is lower than the minimum supported version " +
                 minVersionStr.str());
      return false;
    }
    return true;
  }

//----------------------------------------------------------------------------//

  static herr_t localPrintError( hid_t estack_id, void *stream )
  {
    printf("H5E message -----------------------\n");
    return H5Eprint2(estack_id, static_cast<FILE*>(stream));
  }
  
//----------------------------------------------------------------------------//

} // end of local namespace

//----------------------------------------------------------------------------//
// Partition implementations
//----------------------------------------------------------------------------//

void 
Partition::addScalarLayer(const Layer &layer)
{
  m_scalarLayers.push_back(layer);
}

//----------------------------------------------------------------------------//

void 
Partition::addVectorLayer(const Layer &layer)
{
  m_vectorLayers.push_back(layer);
}

//----------------------------------------------------------------------------//

const Layer* 
Partition::scalarLayer(const std::string &name) const
{
  for (ScalarLayerList::const_iterator i = m_scalarLayers.begin();
       i != m_scalarLayers.end(); ++i) {
    if (i->name == name)
      return &(*i);
  }
  return NULL;
}

//----------------------------------------------------------------------------//

const Layer* 
Partition::vectorLayer(const std::string &name) const
{
  for (VectorLayerList::const_iterator i = m_vectorLayers.begin();
       i != m_vectorLayers.end(); ++i) {
    if (i->name == name)
      return &(*i);
  }
  return NULL;
}

//----------------------------------------------------------------------------//

void 
Partition::getScalarLayerNames(std::vector<std::string> &names) const 
{
  // We don't want to do names.clear() here, since this gets called
  // inside some loops that want to accumulate names.
  for (ScalarLayerList::const_iterator i = m_scalarLayers.begin();
       i != m_scalarLayers.end(); ++i) {
    names.push_back(i->name);
  }
}

//----------------------------------------------------------------------------//

void 
Partition::getVectorLayerNames(std::vector<std::string> &names) const
{
  // We don't want to do names.clear() here, since this gets called
  // inside some loops that want to accumulate names.
  for (VectorLayerList::const_iterator i = m_vectorLayers.begin();
       i != m_vectorLayers.end(); ++i) {
    names.push_back(i->name);
  }
}

//----------------------------------------------------------------------------//
// Field3DFileBase implementations
//----------------------------------------------------------------------------//

Field3DFileBase::Field3DFileBase()
  : m_file(-1), m_metadata(this)
{
  // Suppressing HDF error messages
  // Explanation about the function for the error stack is here:
  // http://www.hdfgroup.org/HDF5/doc/RM/RM_H5E.html#Error-SetAuto2
  if (getenv("DEBUG_HDF")) {
    cerr << "Field3DFile -- HDF5 messages are on" << endl;
    H5Eset_auto(H5E_DEFAULT, localPrintError, NULL);
  } else {
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);
  }
}

//----------------------------------------------------------------------------//

Field3DFileBase::~Field3DFileBase()
{
  close();
}

//----------------------------------------------------------------------------//

std::string 
Field3DFileBase::intPartitionName(const std::string &partitionName,
                                  const std::string &layerName,
                                  FieldRes::Ptr field)
{
  // Loop over existing partitions and see if there's a matching mapping
  for (PartitionList::const_iterator i = m_partitions.begin();
       i != m_partitions.end(); ++i) {
    if (removeUniqueId((**i).name) == partitionName) {
      if ((**i).mapping->isIdentical(field->mapping())) {
        return (**i).name;
      }
    }
  }

  // If there was no previously matching name, then make a new one

  int nextIdx = -1;
  if (m_partitionCount.find(partitionName) != m_partitionCount.end()) {
    nextIdx = ++m_partitionCount[partitionName];
  } else {
    nextIdx = 0;
    m_partitionCount[partitionName] = 0;
  }

  return makeIntPartitionName(partitionName, nextIdx);
}

//----------------------------------------------------------------------------//

Partition::Ptr Field3DFileBase::partition(const string &partitionName) 
{
  for (PartitionList::iterator i = m_partitions.begin();
       i != m_partitions.end(); ++i) {
    if ((**i).name == partitionName)
      return *i;
  }

  return Partition::Ptr();
}

//----------------------------------------------------------------------------//

Partition::Ptr
Field3DFileBase::partition(const string &partitionName) const
{
  for (PartitionList::const_iterator i = m_partitions.begin();
       i != m_partitions.end(); ++i) {
    if ((**i).name == partitionName)
      return *i;
  }

  return Partition::Ptr();
}

//----------------------------------------------------------------------------//

std::string 
Field3DFileBase::removeUniqueId(const std::string &partitionName) const
{
  size_t pos = partitionName.rfind(".");
  if (pos == partitionName.npos) {
    return partitionName;
  } else {
    return partitionName.substr(0, pos);
  }  
}

//----------------------------------------------------------------------------//

void 
Field3DFileBase::getPartitionNames(vector<string> &names) const
{
  names.clear();

  vector<string> tempNames;

  for (PartitionList::const_iterator i = m_partitions.begin();
       i != m_partitions.end(); ++i) {
    tempNames.push_back(removeUniqueId((**i).name));
  }

  names = makeUnique(tempNames);
}

//----------------------------------------------------------------------------//

void 
Field3DFileBase::getScalarLayerNames(vector<string> &names, 
                                     const string &partitionName) const
{
  names.clear();

  for (int i = 0; i < numIntPartitions(partitionName); i++) {
    string internalName = makeIntPartitionName(partitionName, i);
    Partition::Ptr part = partition(internalName);
    if (part)
      part->getScalarLayerNames(names);
  }

  names = makeUnique(names);
}

//----------------------------------------------------------------------------//

void 
Field3DFileBase::getVectorLayerNames(vector<string> &names, 
                                     const string &partitionName) const
{
  names.clear();

  for (int i = 0; i < numIntPartitions(partitionName); i++) {
    string internalName = makeIntPartitionName(partitionName, i);
    Partition::Ptr part = partition(internalName);
    if (part)
      part->getVectorLayerNames(names);
  }

  names = makeUnique(names);
}

//----------------------------------------------------------------------------//

void 
Field3DFileBase::getIntPartitionNames(vector<string> &names) const
{
  names.clear();

  for (PartitionList::const_iterator i = m_partitions.begin();
       i != m_partitions.end(); ++i) {
    names.push_back((**i).name);
  }
}

//----------------------------------------------------------------------------//

void 
Field3DFileBase::getIntScalarLayerNames(vector<string> &names, 
                                        const string &intPartitionName) const
{
  names.clear();

  Partition::Ptr part = partition(intPartitionName);

  if (!part) {
    Msg::print("getIntScalarLayerNames no partition: " + intPartitionName);
    return;
  }

  part->getScalarLayerNames(names);
}

//----------------------------------------------------------------------------//

void 
Field3DFileBase::getIntVectorLayerNames(vector<string> &names, 
                                        const string &intPartitionName) const
{
  names.clear();

  Partition::Ptr part = partition(intPartitionName);

  if (!part) {
    Msg::print("getIntVectorLayerNames no partition: " + intPartitionName);    
    return;
  }

  part->getVectorLayerNames(names);
}

//----------------------------------------------------------------------------//

void Field3DFileBase::clear()
{
  closeInternal();
  m_partitions.clear();
  m_groupMembership.clear();
}

//----------------------------------------------------------------------------//

bool Field3DFileBase::close()
{
  closeInternal();

  return true;
}

//----------------------------------------------------------------------------//

void Field3DFileBase::closeInternal()
{
  if (m_file != -1) {
    if (H5Fclose(m_file) < 0) {
      Msg::print(Msg::SevWarning, "Failed to close hdf5 file handle");
      return;
    }    
    m_file = -1;
  }
}

//----------------------------------------------------------------------------//

int 
Field3DFileBase::numIntPartitions(const std::string &partitionName) const
{
  int count = 0;

  for (PartitionList::const_iterator i = m_partitions.begin();
       i != m_partitions.end(); ++i) {
    string name = (**i).name;
    size_t pos = name.rfind(".");
    if (pos != name.npos) {
      if (name.substr(0, pos) == partitionName) {
        count++;
      }
    }
  }

  return count;
}

//----------------------------------------------------------------------------//

string 
Field3DFileBase::makeIntPartitionName(const std::string &partitionName,
                                      int i) const
{
  return partitionName + "." + boost::lexical_cast<std::string>(i);
}

//----------------------------------------------------------------------------//

void 
Field3DFileBase::addGroupMembership(const GroupMembershipMap& groupMembers)
{
  GroupMembershipMap::const_iterator i= groupMembers.begin();
  GroupMembershipMap::const_iterator end= groupMembers.end();

  for (; i != end; ++i) {
    GroupMembershipMap::iterator foundGroupIter = 
      m_groupMembership.find(i->first);
    if (foundGroupIter != m_groupMembership.end()){
      std::string value = m_groupMembership[i->first] + i->second;
      m_groupMembership[i->first] = value;
    } else { 
      m_groupMembership[i->first] = i->second;
    }
  }
}

//----------------------------------------------------------------------------//
// Field3DInputFile implementations
//----------------------------------------------------------------------------//

Field3DInputFile::Field3DInputFile() 
{ 
  // Empty
}

//----------------------------------------------------------------------------//

Field3DInputFile::~Field3DInputFile() 
{ 
  clear(); 
}

//----------------------------------------------------------------------------//

bool Field3DInputFile::open(const string &filename)
{
  clear();

  bool success = true;

  m_filename = filename;

  try {

    string version;

    // Throws exceptions if the file doesn't exist.
    // This was added because H5Fopen prints out a lot of junk
    // to the terminal.
    checkFile(filename);

    m_file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    if (m_file < 0)
      throw NoSuchFileException(filename);

    int fileVersion[3];
    try { 
      if (!readAttribute(m_file, k_versionAttrName, 3, fileVersion[0])) {
        //Msg::print(Msg::SevWarning, "Missing version_number attribute");
      } else {
        if (!isSupportedFileVersion(fileVersion, k_minFileVersion)) {
          stringstream versionStr;
          versionStr << fileVersion[0] << "."
                     << fileVersion[1] << "."
                     << fileVersion[2];
          throw UnsupportedVersionException(versionStr.str());
        }
      }
    }
    catch (MissingAttributeException &e) {
      //Msg::print(Msg::SevWarning, "Missing version_number attribute");
    }

    try { 
      if (H5Lexists(m_file, "field3d_global_metadata", H5P_DEFAULT)) {      
        // read the metadata 
        H5ScopedGopen metadataGroup(m_file, "field3d_global_metadata");
        if (metadataGroup.id() > 0) {    
          readMetadata(metadataGroup.id());
        }
      }
    }
    catch (...) {
      Msg::print(Msg::SevWarning, 
                 "Unknown error when reading file metadata ");
      //throw BadFileHierarchyException(filename);
    }

    try {
      if (!readPartitionAndLayerInfo()) {
        success = false;
      }
    } 
    catch (MissingGroupException &e) {
      Msg::print(Msg::SevWarning, "Missing group: " + string(e.what()));
      throw BadFileHierarchyException(filename);
    }
    catch (ReadMappingException &e) {
      Msg::print(Msg::SevWarning, "Couldn't read mapping for partition: " 
                + string(e.what()));
      throw BadFileHierarchyException(filename);
    }
    catch (Exception &e) {
      Msg::print(Msg::SevWarning, "Unknown error when reading file hierarchy: "
                + string(e.what()));
      throw BadFileHierarchyException(filename);
    }
    catch (...) {
      Msg::print(Msg::SevWarning, 
                 "Unknown error when reading file hierarchy. ");
      throw BadFileHierarchyException(filename);
    }

  }
  catch (NoSuchFileException &e) {
    Msg::print(Msg::SevWarning, "Couldn't open file: " 
              + string(e.what()) );
    success = false;    
  }
  catch (MissingAttributeException &e) {
    Msg::print(Msg::SevWarning, 
               "In file: " + filename + " - "
              + string(e.what()) );
    success = false;
  }
  catch (UnsupportedVersionException &e) {    
    Msg::print(Msg::SevWarning, 
               "In file: " + filename + " - File version can not be read: " 
              + string(e.what()));
    success = false;    
  }
  catch (BadFileHierarchyException &e) {
    Msg::print(Msg::SevWarning, 
               "In file: " + filename + " - Bad file hierarchy. ");
    success = false;    
  }
  catch (...) {
    Msg::print(Msg::SevWarning, 
               "In file: " + filename + " Unknown exception ");
    success = false;
  }

  if (!success)
    close();
  
  return success;
}

//----------------------------------------------------------------------------//

bool Field3DInputFile::readPartitionAndLayerInfo()
{
  using namespace InputFile;

  // First, find the partitions ---

  herr_t status;
  status = H5Literate(m_file, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, 
                      &parsePartitions, this);

  // Get the partition names to store 
  m_partitions.clear();

  for (size_t i=0; i < m_partitionNames.size(); i++) {
    Partition::Ptr part(new Partition);
    part->name = m_partitionNames[i];    
    m_partitions.push_back(part);
  }
  
  // For each partition, find its mapping ---

  for (PartitionList::iterator i = m_partitions.begin();
       i != m_partitions.end(); ++i) {

    // Open the partition
    H5ScopedGopen partitionGroup(m_file, (**i).name);

    string mappingPath = "/" + (**i).name + "/" + k_mappingStr;

    // Open up the mapping group
    H5ScopedGopen mappingGroup(m_file, mappingPath);
    if (mappingGroup.id() < 0)
      throw MissingGroupException((**i).name + "/" + k_mappingStr);

    // Try to build a mapping from it
    FieldMapping::Ptr mapping;

    mapping = readFieldMapping(mappingGroup.id());
    if (!mapping) {
      Msg::print(Msg::SevWarning, "Got a null pointer when reading mapping");
      throw ReadMappingException((**i).name);
    }
    
    // Attach the mapping to the partition
    (**i).mapping = mapping;

  }

  // ... And then find its layers ---

  for (PartitionList::const_iterator i = m_partitions.begin();
       i != m_partitions.end(); ++i) {

    // Open the partition
    H5ScopedGopen partitionGroup(m_file, (**i).name);
    
    // Set up the info struct for the callback
    ParseLayersInfo info;
    info.file = this;
    info.partitionName = (**i).name;

    m_layerInfo.clear();

    status = H5Literate(partitionGroup.id(), H5_INDEX_NAME, H5_ITER_NATIVE, 
                        NULL, &parseLayers, &info);

    //set the layer information on the partitions here

    for (std::vector<LayerInfo>::iterator i = m_layerInfo.begin();
         i != m_layerInfo.end(); i++) {

      std::string parent = i->parentName;      

      Partition::Ptr part = partition(parent);

      Layer layer;
      layer.name = i->name;
      layer.parent = i->parentName;
      if (i->components == 1) {
        part->addScalarLayer(layer);
      } else if (i->components == 3) {
        part->addVectorLayer(layer);
      }
    }

  }

  return true;
}

//----------------------------------------------------------------------------//

herr_t Field3DInputFile::parsePartition(hid_t loc_id, 
                                        const std::string itemName)
{
  // Add the partition ---
  
  m_partitionNames.push_back(string(itemName));
  return 0;
}

//----------------------------------------------------------------------------//

//! \note Don't throw exceptions into the hdf5 lib.
//! \todo Set some sort of flag if we fail during this call. We can't
//! throw exceptions inside hdf5.
herr_t Field3DInputFile::parseLayer(hid_t layerGroup, 
                               const std::string &partitionName,
                               const std::string &layerName)
{
  int components;
  if (!readAttribute(layerGroup, string("components"), 1, components)) {
    Msg::print(Msg::SevWarning, "Couldn't read components attribute for layer " 
              + partitionName + "/" + layerName);
    return 0;
  }

  LayerInfo linfo(partitionName,layerName,components);

  m_layerInfo.push_back(linfo);

  return 0;
}

//----------------------------------------------------------------------------//

//! \todo Replace char* with std::string
bool  
Field3DInputFile::
readMetadata(hid_t metadata_id, FieldBase::Ptr field) const
{

  hsize_t num_attrs = H5Aget_num_attrs(metadata_id);

  if (num_attrs > 0) { 
    for (hsize_t idx=0; idx < num_attrs ; ++idx) {
      H5ScopedAopenIdx attrIdx(metadata_id, idx);
      size_t len = H5Aget_name(attrIdx.id(), 0, NULL);
      if (len > 0) {
        char *name = new char[len+1];
        if (H5Aget_name(attrIdx.id(), len+1, name) > 0) {
          H5ScopedAopen attr(metadata_id, name, H5P_DEFAULT);
          H5ScopedAget_space attrSpace(attr);
          H5ScopedAget_type attrType(attr);           
          H5T_class_t typeClass = H5Tget_class(attrType);

          if (typeClass == H5T_STRING) { 
            string value;
            if (!readAttribute(metadata_id, name, value)) {
              Msg::print(Msg::SevWarning, 
                         "Failed to read metadata " + string(name));
              if (name) {
                delete[] name;
              }
              continue;
            }
            field->metadata().setStrMetadata(name, value);
             
          }
          else {

            if (H5Sget_simple_extent_ndims(attrSpace) != 1) {
              Msg::print(Msg::SevWarning, "Bad attribute rank for attribute " 
                        + string(name));
              if (name) {
                delete[] name;
              }
              continue;
            }            

            hsize_t dims[1];
            H5Sget_simple_extent_dims(attrSpace, dims, NULL);
 
            if (typeClass == H5T_INTEGER) { 
              if (dims[0] == 1){
                int value;
                if (!readAttribute(metadata_id, name, dims[0], value))
                  Msg::print(Msg::SevWarning, "Failed to read metadata " 
                            + string(name));
                field->metadata().setIntMetadata(name, value);
              }
              else if (dims[0] == 3){
                V3i value;
                if (!readAttribute(metadata_id, name, dims[0], value.x))
                  Msg::print(Msg::SevWarning, "Failed to read metadata " + 
                            string(name) );
                field->metadata().setVecIntMetadata(name, value);
              }
              else {
                Msg::print(Msg::SevWarning, 
                           "Attribute of size " + 
                           boost::lexical_cast<std::string>(dims[0]) 
                           + " is not valid for metadata");
              }
            }
            else if (typeClass == H5T_FLOAT) { 
              if (dims[0] == 1){
                float value;
                if (!readAttribute(metadata_id, name, dims[0], value))
                  Msg::print(Msg::SevWarning, "Failed to read metadata " + 
                            string(name) );
                
                field->metadata().setFloatMetadata(name, value);
              }
              else if (dims[0] == 3){
                V3f value;
                if (!readAttribute(metadata_id, name, dims[0], value.x))
                  Msg::print(Msg::SevWarning, "Failed to read metadata "+ 
                            string(name) );
                field->metadata().setVecFloatMetadata(name, value);
              }
              else {
                Msg::print(Msg::SevWarning, "Attribute of size " +
                           boost::lexical_cast<std::string>(dims[0]) +
                           " is not valid for metadata");
              }
            }
            else {               
              Msg::print(Msg::SevWarning, "Attribute '" + string(name) + 
                        + "' has unsupported data type for metadata");
              
            }
          }
        }
        if (name) {
          delete[] name;
        }
      }
    }
  }

  return true;
}

//----------------------------------------------------------------------------//

//! \todo Replace char* with std::string
bool  
Field3DInputFile::readMetadata(hid_t metadata_id)
{

  hsize_t num_attrs = H5Aget_num_attrs(metadata_id);

  if (num_attrs > 0) { 
    for (hsize_t idx=0; idx < num_attrs ; ++idx) {
      H5ScopedAopenIdx attrIdx(metadata_id, idx);
      size_t len = H5Aget_name(attrIdx.id(), 0, NULL);
      if (len > 0) {
        char *name = new char[len+1];
        if (H5Aget_name(attrIdx.id(), len+1, name) > 0) {
          H5ScopedAopen attr(metadata_id, name, H5P_DEFAULT);
          H5ScopedAget_space attrSpace(attr);
          H5ScopedAget_type attrType(attr);           
          H5T_class_t typeClass = H5Tget_class(attrType);

          if (typeClass == H5T_STRING) { 
            string value;
            if (!readAttribute(metadata_id, name, value)) {
              Msg::print(Msg::SevWarning, 
                         "Failed to read metadata " + string(name));
              if (name) {
                delete[] name;
              }
              continue;
            }
            metadata().setStrMetadata(name, value);
             
          }
          else {

            if (H5Sget_simple_extent_ndims(attrSpace) != 1) {
              Msg::print(Msg::SevWarning, "Bad attribute rank for attribute " 
                        + string(name));
              if (name) {
                delete[] name;
              }
              continue;
            }            

            hsize_t dims[1];
            H5Sget_simple_extent_dims(attrSpace, dims, NULL);
 
            if (typeClass == H5T_INTEGER) { 
              if (dims[0] == 1){
                int value;
                if (!readAttribute(metadata_id, name, dims[0], value))
                  Msg::print(Msg::SevWarning, "Failed to read metadata " 
                            + string(name));
                metadata().setIntMetadata(name, value);
              }
              else if (dims[0] == 3){
                V3i value;
                if (!readAttribute(metadata_id, name, dims[0], value.x))
                  Msg::print(Msg::SevWarning, "Failed to read metadata " + 
                            string(name) );
                metadata().setVecIntMetadata(name, value);
              }
              else {
                Msg::print(Msg::SevWarning, 
                           "Attribute of size " + 
                           boost::lexical_cast<std::string>(dims[0]) 
                           + " is not valid for metadata");
              }
            }
            else if (typeClass == H5T_FLOAT) { 
              if (dims[0] == 1){
                float value;
                if (!readAttribute(metadata_id, name, dims[0], value))
                  Msg::print(Msg::SevWarning, "Failed to read metadata " + 
                            string(name) );
                
                metadata().setFloatMetadata(name, value);
              }
              else if (dims[0] == 3){
                V3f value;
                if (!readAttribute(metadata_id, name, dims[0], value.x))
                  Msg::print(Msg::SevWarning, "Failed to read metadata "+ 
                            string(name) );
                metadata().setVecFloatMetadata(name, value);
              }
              else {
                Msg::print(Msg::SevWarning, "Attribute of size " +
                           boost::lexical_cast<std::string>(dims[0]) +
                           " is not valid for metadata");
              }
            }
            else {               
              Msg::print(Msg::SevWarning, "Attribute '" + string(name) + 
                        + "' has unsupported data type for metadata");
              
            }
          }
        }
        if (name) {
          delete[] name;
        }
      }
    }
  }

  return true;
}

//----------------------------------------------------------------------------//

bool
Field3DInputFile::
readGroupMembership(GroupMembershipMap &gpMembershipMap)
{
  if (!H5Lexists(m_file, "field3d_group_membership", H5P_DEFAULT)) {
    return false;
  }

  H5ScopedGopen memberGroup(m_file, "field3d_group_membership");
  if (memberGroup < 0) {
    return false;
  }
  
  typedef boost::tokenizer<boost::char_separator<char> > Tok;

  hsize_t num_attrs = H5Aget_num_attrs(memberGroup);
  if (num_attrs > 0) { 
    
    for (hsize_t idx=0; idx < num_attrs ; ++idx) {
      H5ScopedAopenIdx attrIdx(memberGroup, idx);        
      size_t len = H5Aget_name(attrIdx.id(), 0, NULL);
      if (len>0) {
        char *name = new char[len+1];
        if (H5Aget_name(attrIdx.id(), len+1, name) > 0) {

          if (string(name) == "is_field3d_group_membership")
            continue;

          H5ScopedAopen attr(memberGroup, name, H5P_DEFAULT);
          H5ScopedAget_space attrSpace(attr);
          H5ScopedAget_type attrType(attr);           
          H5T_class_t typeClass = H5Tget_class(attrType);

          if (typeClass == H5T_STRING) { 
            string value;
            if (!readAttribute(memberGroup, name, value)) {
              Msg::print(Msg::SevWarning, 
                         "Failed to read group membership data  " 
                        + string(name));
              continue;
            }

            {
              boost::char_separator<char> sep(" :");
              Tok tok(value, sep);
              string new_value;
              for(Tok::iterator beg=tok.begin(); beg!=tok.end();){

                string fieldgroup = *beg; ++beg;
                fieldgroup = removeUniqueId(fieldgroup) + ":" + *beg; ++beg;
                new_value += fieldgroup + " "; 
              }

              m_groupMembership[name] = value;
              gpMembershipMap[name] = new_value;
            }
          }
        }
      }
    }
  }

  return true;
}


//----------------------------------------------------------------------------//
// Field3DFile-related callback functions
//----------------------------------------------------------------------------//

namespace InputFile {

//----------------------------------------------------------------------------//

herr_t parsePartitions(hid_t loc_id, const char *itemName, 
                       const H5L_info_t *linfo, void *opdata)
{
  herr_t          status;
  H5O_info_t      infobuf;

  status = H5Oget_info_by_name(loc_id, itemName, &infobuf, H5P_DEFAULT);

  if (status < 0) {
    return -1;
  }
  
  if (infobuf.type == H5O_TYPE_GROUP) {

    // Check that we have a name 
    if (!itemName) {
      return -1;
    }

    // check that this group is not "groupMembership"
    if (string(itemName) != "field3d_group_membership" &&
        string(itemName) != "field3d_global_metadata")
    { 

      // Get a pointer to the file data structure
      Field3DInputFile* fileObject = static_cast<Field3DInputFile*>(opdata);
      if (!fileObject) {
        return -1;
      }
      
      return fileObject->parsePartition(loc_id, itemName);
    }
  }
  return 0;
}

//----------------------------------------------------------------------------//

herr_t parseLayers(hid_t loc_id, const char *itemName, 
                   const H5L_info_t *linfo, void *opdata)
{
  herr_t          status;
  H5O_info_t      infobuf;
  
  status = H5Oget_info_by_name (loc_id, itemName, &infobuf, H5P_DEFAULT);

  if (infobuf.type == H5O_TYPE_GROUP) {

    // Check that we have a name 
    if (!itemName)
      return -1;

    // Get a pointer to the file data structure
    ParseLayersInfo* info = static_cast<ParseLayersInfo*>(opdata);
    if (!info) 
      return -1;

    // Open up the layer group
    H5ScopedGopen layerGroup(loc_id, itemName);

    // Check if it's a layer
    string classType;
    try {
      if (!readAttribute(layerGroup.id(), "class_type", classType)) {
        return 0;
      }
      if (classType == string("field3d_layer")) 
        return info->file->parseLayer(layerGroup.id(), info->partitionName,
                                      itemName);
                                   
    }
    catch (MissingAttributeException &e) {
      
    }
    return 0;

  }

  return 0;
}

//----------------------------------------------------------------------------//

} // namespace InputFile

//----------------------------------------------------------------------------//
// Field3DOutputFile implementations
//----------------------------------------------------------------------------//

Field3DOutputFile::Field3DOutputFile() 
{ 
  // Empty
}

//----------------------------------------------------------------------------//

Field3DOutputFile::~Field3DOutputFile() 
{ 

}

//----------------------------------------------------------------------------//

//! \todo If the file can't be created hdf5 spits out an ugly error msg,
//! we should make sure that doesn't happen.
bool Field3DOutputFile::create(const string &filename, CreateMode cm)
{
  closeInternal();

  bool success = true;

  try {

    hid_t faid = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(faid, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);

    // Create new file
    switch (cm) {
    case OverwriteMode:
      m_file = H5Fcreate(filename.c_str(), 
                         H5F_ACC_TRUNC, H5P_DEFAULT, faid);
      break;
    case FailOnExisting:
      m_file = H5Fcreate(filename.c_str(), 
                         H5F_ACC_EXCL, H5P_DEFAULT, faid);
      break;
    }
    
    // Check that file was created
    if (m_file < 0)
      throw ErrorCreatingFileException(filename);
    
    // Create a version attribute on the root node
    if (!writeAttribute(m_file, k_versionAttrName, 3,
                        k_currentFileVersion[0])) {
      Msg::print(Msg::SevWarning, "Adding version number.");
      closeInternal();
      return false;
    }

  }
  catch (ErrorCreatingFileException &e) {
    Msg::print(Msg::SevWarning, "Couldn't create file: " + string(e.what()) );
    success = false;
  } 
  catch (WriteAttributeException &e) {
    Msg::print(Msg::SevWarning, "In file : " + filename +
              " - Couldn't add attribute " + string(e.what()) );
    success = false;
  }
  catch (...) {
    Msg::print(Msg::SevWarning, 
               "Unknown error when creating file: " + filename );
    success = false;
  }

  return success;
}

//----------------------------------------------------------------------------//

bool Field3DOutputFile::writeMapping(hid_t partitionGroup, 
                                     FieldMapping::Ptr mapping)
{
  try {
    // Make a group under the partition to store the mapping data
    H5ScopedGcreate mappingGroup(partitionGroup, k_mappingStr);
    if (mappingGroup.id() < 0)
      throw CreateGroupException(k_mappingStr);
    // Let FieldMappingIO handle the rest
    if (!writeFieldMapping(mappingGroup.id(), mapping))
      throw WriteMappingException(k_mappingStr);       
  }
  catch (CreateGroupException &e) {
    Msg::print(Msg::SevWarning, "Couldn't create group: " + string(e.what()) );
    throw WriteMappingException(k_mappingStr);
  }
  return true;
}

//----------------------------------------------------------------------------//

bool Field3DOutputFile::writeMetadata(hid_t metadataGroup, FieldBase::Ptr field)
{
  using namespace Hdf5Util;

  {
    FieldMetadata<FieldBase>::StrMetadata::const_iterator i = 
      field->metadata().strMetadata().begin();
    FieldMetadata<FieldBase>::StrMetadata::const_iterator end = 
      field->metadata().strMetadata().end();
    for (; i != end; ++i) {
      if (!writeAttribute(metadataGroup, i->first, i->second))
      {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first );
        return false;
      }
    }
  }

  {
    FieldMetadata<FieldBase>::IntMetadata::const_iterator i = 
      field->metadata().intMetadata().begin();
    FieldMetadata<FieldBase>::IntMetadata::const_iterator end = 
      field->metadata().intMetadata().end();
    for (; i != end; ++i) {
      if (!writeAttribute(metadataGroup, i->first, 1, i->second))
      {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first);
        return false;
      }
    }
  }

  {
    FieldMetadata<FieldBase>::FloatMetadata::const_iterator i = 
      field->metadata().floatMetadata().begin();
    FieldMetadata<FieldBase>::FloatMetadata::const_iterator end = 
      field->metadata().floatMetadata().end();
    for (; i != end; ++i) {
      if (!writeAttribute(metadataGroup, i->first, 1, i->second))
      {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first);
        return false;
      }
    }
  }

  {
    FieldMetadata<FieldBase>::VecIntMetadata::const_iterator i = 
      field->metadata().vecIntMetadata().begin();
    FieldMetadata<FieldBase>::VecIntMetadata::const_iterator end = 
      field->metadata().vecIntMetadata().end();
    for (; i != end; ++i) {
      if (!writeAttribute(metadataGroup, i->first, 3, i->second.x))
      {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first);
        return false;
      }
    }
  }

  {
    FieldMetadata<FieldBase>::VecFloatMetadata::const_iterator i = 
      field->metadata().vecFloatMetadata().begin();
    FieldMetadata<FieldBase>::VecFloatMetadata::const_iterator end = 
      field->metadata().vecFloatMetadata().end();
    for (; i != end; ++i) {
      if (!writeAttribute(metadataGroup, i->first, 3, i->second.x))
      {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first);
        return false;
      }
    }

  }

  return true;

}

//----------------------------------------------------------------------------//

bool Field3DOutputFile::writeMetadata(hid_t metadataGroup)
{
  using namespace Hdf5Util;

  {
    FieldMetadata<Field3DFileBase>::StrMetadata::const_iterator i = 
      metadata().strMetadata().begin();
    FieldMetadata<Field3DFileBase>::StrMetadata::const_iterator end = 
      metadata().strMetadata().end();
    for (; i != end; ++i) {
      if (!writeAttribute(metadataGroup, i->first, i->second))
      {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first );
        return false;
      }
    }
  }

  {
    FieldMetadata<Field3DFileBase>::IntMetadata::const_iterator i = 
      metadata().intMetadata().begin();
    FieldMetadata<Field3DFileBase>::IntMetadata::const_iterator end = 
      metadata().intMetadata().end();
    for (; i != end; ++i) {
      if (!writeAttribute(metadataGroup, i->first, 1, i->second))
      {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first);
        return false;
      }
    }
  }

  {
    FieldMetadata<Field3DFileBase>::FloatMetadata::const_iterator i = 
      metadata().floatMetadata().begin();
    FieldMetadata<Field3DFileBase>::FloatMetadata::const_iterator end = 
      metadata().floatMetadata().end();
    for (; i != end; ++i) {
      if (!writeAttribute(metadataGroup, i->first, 1, i->second))
      {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first);
        return false;
      }
    }
  }

  {
    FieldMetadata<Field3DFileBase>::VecIntMetadata::const_iterator i = 
      metadata().vecIntMetadata().begin();
    FieldMetadata<Field3DFileBase>::VecIntMetadata::const_iterator end = 
      metadata().vecIntMetadata().end();
    for (; i != end; ++i) {
      if (!writeAttribute(metadataGroup, i->first, 3, i->second.x))
      {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first);
        return false;
      }
    }
  }

  {
    FieldMetadata<Field3DFileBase>::VecFloatMetadata::const_iterator i = 
      metadata().vecFloatMetadata().begin();
    FieldMetadata<Field3DFileBase>::VecFloatMetadata::const_iterator end = 
      metadata().vecFloatMetadata().end();
    for (; i != end; ++i) {
      if (!writeAttribute(metadataGroup, i->first, 3, i->second.x))
      {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first);
        return false;
      }
    }

  }

  return true;

}

//----------------------------------------------------------------------------//

bool 
Field3DOutputFile::writeGlobalMetadata()
{

  // Add metadata group and write it out  
  H5ScopedGcreate metadataGroup(m_file, "field3d_global_metadata");
  if (metadataGroup.id() < 0) {
    Msg::print(Msg::SevWarning, "Error creating group: file metadata");
    return false;
  }  
  if (!writeMetadata(metadataGroup.id())) {
    Msg::print(Msg::SevWarning, "Error writing file metadata.");
    return false;
  }    
 
  return true;
}

//----------------------------------------------------------------------------//

bool 
Field3DOutputFile::writeGroupMembership()
{
  using namespace std;
  using namespace Hdf5Util;

  if (!m_groupMembership.size())
    return true;

  H5ScopedGcreate group(m_file, "field3d_group_membership");
  if (group < 0) {
    Msg::print(Msg::SevWarning, 
               "Error creating field3d_group_membership group.");      
    return false;
  } 

  if (!writeAttribute(group, "is_field3d_group_membership", "1")) {
    Msg::print(Msg::SevWarning, 
               "Failed to write field3d_group_membership attribute.");
    return false;
  }    

  std::map<std::string, std::string>::const_iterator iter = 
    m_groupMembership.begin();
  std::map<std::string, std::string>::const_iterator iEnd = 
    m_groupMembership.end();
  
  for (; iter != iEnd; ++iter) {
    if (!writeAttribute(group, iter->first, iter->second)) {
      Msg::print(Msg::SevWarning, 
                 "Failed to write groupMembership string: "+ iter->first);
      return false;
    }        
  }
  
  return true;
}

//----------------------------------------------------------------------------//
// Debug
//----------------------------------------------------------------------------//

void Field3DFileBase::printHierarchy() const
{
  // For each partition
  for (PartitionList::const_iterator i = m_partitions.begin();
       i != m_partitions.end(); ++i) {
    cout << "Name: " << (**i).name << endl;
    if ((**i).mapping)
      cout << "  Mapping: " << (**i).mapping->className() << endl;
    else 
      cout << "  Mapping: NULL" << endl;
    cout << "  Scalar layers: " << endl;
    vector<string> sNames;
    (**i).getScalarLayerNames(sNames);
    for_each(sNames.begin(), sNames.end(), print<string>(4));
    cout << "  Vector layers: " << endl;
    vector<string> vNames;
    (**i).getVectorLayerNames(vNames);
    for_each(vNames.begin(), vNames.end(), print<string>(4));
  }
}

//----------------------------------------------------------------------------//
// Function Implementations
//----------------------------------------------------------------------------//

bool writeField(hid_t layerGroup, FieldBase::Ptr field)
{
  ClassFactory &factory = ClassFactory::singleton();
    
  FieldIO::Ptr io = factory.createFieldIO(field->className());
  assert(io != 0);
  if (!io) {
    Msg::print(Msg::SevWarning, "Unable to find class type: " + 
               field->className());
    return false;
  }

  // Add class name attribute
  if (!writeAttribute(layerGroup, k_classNameAttrName, 
                      field->className())) {
    Msg::print(Msg::SevWarning, "Error adding class name attribute.");
    return false;
  }

  return io->write(layerGroup, field);
}

//----------------------------------------------------------------------------//

FieldMapping::Ptr readFieldMapping(hid_t mappingGroup)
{
  ClassFactory &factory = ClassFactory::singleton();

  std::string className;

  if (!readAttribute(mappingGroup, k_mappingTypeAttrName, className)) {
    Msg::print(Msg::SevWarning, "Couldn't find " + k_mappingTypeAttrName + 
              " attribute");
    return FieldMapping::Ptr();    
  }

  FieldMappingIO::Ptr io = factory.createFieldMappingIO(className);
  assert(io != 0);
  if (!io) {
    Msg::print(Msg::SevWarning, "Unable to find class type: " + 
               className);
    return FieldMapping::Ptr();
  }


  FieldMapping::Ptr mapping = io->read(mappingGroup);
  if (!mapping) {
    Msg::print(Msg::SevWarning, "Couldn't read mapping");
    return FieldMapping::Ptr();
  }
  
  return mapping;
}

//----------------------------------------------------------------------------//

bool writeFieldMapping(hid_t mappingGroup, FieldMapping::Ptr mapping)
{
  ClassFactory &factory = ClassFactory::singleton();

  std::string className = mapping->className();

  if (!writeAttribute(mappingGroup, k_mappingTypeAttrName, className)) {
    Msg::print(Msg::SevWarning, "Couldn't add " + className + " attribute");
    return false;
  }

  FieldMappingIO::Ptr io = factory.createFieldMappingIO(className);
  assert(io != 0);
  if (!io) {
    Msg::print(Msg::SevWarning, "Unable to find class type: " + 
               className);
    return false;
  }

  return io->write(mappingGroup, mapping);
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
