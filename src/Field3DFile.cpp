//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2014 Sony Pictures Imageworks Inc., 
 *                    Pixar Animation Studios Inc.
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

#include "Field3DFile.h"

#include <sys/stat.h>
#ifndef WIN32
#include <unistd.h>
#endif

#include <boost/tokenizer.hpp>
#include <boost/utility.hpp>

#include "Field.h"
#include "FieldCache.h"
#include "Field3DFileHDF5.h"
#include "ClassFactory.h"
#include "OArchive.h"
#include "OgIAttribute.h"
#include "OgIDataset.h"
#include "OgIGroup.h"
#include "OgOAttribute.h"
#include "OgODataset.h"
#include "OgOGroup.h"

//----------------------------------------------------------------------------//

using namespace std;

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Field3D namespaces
//----------------------------------------------------------------------------//

using namespace Exc;

//----------------------------------------------------------------------------//
// Local namespace
//----------------------------------------------------------------------------//

namespace {
  
  // Strings used only in this file --------------------------------------------

  const std::string k_mappingStr("mapping");
  const std::string k_partitionName("partition");  
  const std::string k_versionAttrName("version_number");
  const std::string k_classNameAttrName("class_name");
  const std::string k_mappingTypeAttrName("mapping_type");

  //! This version is stored in every file to determine which library version
  //! produced it.

  V3i k_currentFileVersion = V3i(FIELD3D_MAJOR_VER, 
                                 FIELD3D_MINOR_VER, 
                                 FIELD3D_MICRO_VER);
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

  //--------------------------------------------------------------------------//

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

  //--------------------------------------------------------------------------//

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

  //--------------------------------------------------------------------------//

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

  //! This function creates a FieldMappingIO instance based on className 
  //! read from mappingGroup location which then reads FieldMapping data
  FIELD3D_API FieldMapping::Ptr readFieldMapping(const OgIGroup &mappingGroup)
  {
    ClassFactory &factory = ClassFactory::singleton();
    
    OgIAttribute<string> mappingAttr = 
      mappingGroup.findAttribute<string>(k_mappingTypeAttrName);
    if (!mappingAttr.isValid()) {
      Msg::print(Msg::SevWarning, "Couldn't find " + k_mappingTypeAttrName + 
                 " attribute");
      return FieldMapping::Ptr();    
    }
    const std::string className = mappingAttr.value();

    FieldMappingIO::Ptr io = factory.createFieldMappingIO(className);
    assert(io != 0);
    if (!io) {
      Msg::print(Msg::SevWarning, "Unable to find class type: " + className);
      return FieldMapping::Ptr();
    }

    FieldMapping::Ptr mapping = io->read(mappingGroup);
    if (!mapping) {
      Msg::print(Msg::SevWarning, "Couldn't read mapping");
      return FieldMapping::Ptr();
    }
  
    return mapping;
  }

  //--------------------------------------------------------------------------//

  //! This function creates a FieldIO instance based on field->className()
  //! which then writes the field data in layerGroup location
  FIELD3D_API bool writeField(OgOGroup &layerGroup, FieldBase::Ptr field)
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
    OgOAttribute<string>(layerGroup, k_classNameAttrName, field->className());

    return io->write(layerGroup, field);
    //! \todo FIXME!
    return false;
  }

  //--------------------------------------------------------------------------//

  //! This function creates a FieldIO instance based on className
  //! which then reads the field data from layerGroup location
  template <class Data_T>
  typename Field<Data_T>::Ptr 
  readField(const std::string &className, const OgIGroup &layerGroup,
            const std::string &filename, const std::string &layerPath)
  {
    ClassFactory &factory = ClassFactory::singleton();
  
    typedef typename Field<Data_T>::Ptr FieldPtr;

    FieldIO::Ptr io = factory.createFieldIO(className);
    if (!io) {
      Msg::print(Msg::SevWarning, "Unable to find class type: " + 
                 className);
      return FieldPtr();
    }

    OgDataType typeEnum = OgawaTypeTraits<Data_T>::typeEnum();
    FieldBase::Ptr field = io->read(layerGroup, filename, layerPath, typeEnum);

    if (!field) {
      // We don't need to print a message, because it could just be that
      // a layer of the specified data type and name couldn't be found
      return FieldPtr();
    }
  
    FieldPtr result = field_dynamic_cast<Field<Data_T> >(field);

    if (result) {
      return result;
    }

    return FieldPtr();
  }

  //--------------------------------------------------------------------------//

  bool readMeta(const OgIGroup &group, FieldMetadata &metadata)
  {
    // Grab all the attribute names
    std::vector<std::string> attrs = group.attributeNames();
    // Loop over attribute names and test types
    for (size_t i = 0, end = attrs.size(); i < end; ++i) {
      // String metadata
      {
        OgIAttribute<string> attr = group.findAttribute<string>(attrs[i]);
        if (attr.isValid()) {
          metadata.setStrMetadata(attrs[i], attr.value());
        }
      }
      // Int metadata
      {
        OgIAttribute<int> attr = group.findAttribute<int>(attrs[i]);
        if (attr.isValid()) {
          metadata.setIntMetadata(attrs[i], attr.value());
        }
      }
      // Float metadata
      {
        OgIAttribute<float> attr = group.findAttribute<float>(attrs[i]);
        if (attr.isValid()) {
          metadata.setFloatMetadata(attrs[i], attr.value());
        }
      }
      // VecInt metadata
      {
        OgIAttribute<veci32_t> attr = group.findAttribute<veci32_t>(attrs[i]);
        if (attr.isValid()) {
          metadata.setVecIntMetadata(attrs[i], attr.value());
        }
      }
      // VecFloat metadata
      {
        OgIAttribute<vec32_t> attr = group.findAttribute<vec32_t>(attrs[i]);
        if (attr.isValid()) {
          metadata.setVecFloatMetadata(attrs[i], attr.value());
        }
      }
    }

    return true;
  }

  //--------------------------------------------------------------------------//

} // end of local namespace

//----------------------------------------------------------------------------//
// File namespace
//----------------------------------------------------------------------------//

namespace File {

//----------------------------------------------------------------------------//
// Partition implementations
//----------------------------------------------------------------------------//

std::string Partition::className() const
{
  return k_partitionName;
}

//----------------------------------------------------------------------------//

void 
Partition::addLayer(const Layer &layer)
{
  m_layers.push_back(layer);
}

//----------------------------------------------------------------------------//

const Layer* 
Partition::layer(const std::string &name) const
{
  for (LayerList::const_iterator i = m_layers.begin(); 
       i != m_layers.end(); ++i) {
    if (i->name == name) {
      return &(*i);
    }
  }
  return NULL;
}

//----------------------------------------------------------------------------//

void 
Partition::getLayerNames(std::vector<std::string> &names) const 
{
  // We don't want to do names.clear() here, since this gets called
  // inside some loops that want to accumulate names.
  for (LayerList::const_iterator i = m_layers.begin();
       i != m_layers.end(); ++i) {
    names.push_back(i->name);
  }
}

//----------------------------------------------------------------------------//

OgOGroup& Partition::group() const
{
  return *m_group;
}

//----------------------------------------------------------------------------//

void Partition::setGroup(boost::shared_ptr<OgOGroup> ptr)
{
  m_group = ptr;
}

//----------------------------------------------------------------------------//

} // namespace File

//----------------------------------------------------------------------------//
// Field3DFileBase implementations
//----------------------------------------------------------------------------//

Field3DFileBase::Field3DFileBase()
  : m_metadata(this)
{
  // Empty
}

//----------------------------------------------------------------------------//

Field3DFileBase::~Field3DFileBase()
{
  m_partitions.clear();
  m_groupMembership.clear();
}

//----------------------------------------------------------------------------//

std::string 
Field3DFileBase::intPartitionName(const std::string &partitionName,
                                  const std::string & /* layerName */,
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

File::Partition::Ptr Field3DFileBase::partition(const string &partitionName) 
{
  for (PartitionList::iterator i = m_partitions.begin();
       i != m_partitions.end(); ++i) {
    if ((**i).name == partitionName)
      return *i;
  }

  return File::Partition::Ptr();
}

//----------------------------------------------------------------------------//

File::Partition::Ptr
Field3DFileBase::partition(const string &partitionName) const
{
  for (PartitionList::const_iterator i = m_partitions.begin();
       i != m_partitions.end(); ++i) {
    if ((**i).name == partitionName)
      return *i;
  }

  return File::Partition::Ptr();
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
  if (m_hdf5Base) {
    m_hdf5Base->getPartitionNames(names);
    return;
  }

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
  if (m_hdf5Base) {
    m_hdf5Base->getScalarLayerNames(names, partitionName);
    return;
  }

  //! \todo Make this really only return scalar layers

  names.clear();

  for (int i = 0; i < numIntPartitions(partitionName); i++) {
    string internalName = makeIntPartitionName(partitionName, i);
    File::Partition::Ptr part = partition(internalName);
    if (part)
      part->getLayerNames(names);
  }

  names = makeUnique(names);
}

//----------------------------------------------------------------------------//

void 
Field3DFileBase::getVectorLayerNames(vector<string> &names, 
                                     const string &partitionName) const
{
  if (m_hdf5Base) {
    m_hdf5Base->getVectorLayerNames(names, partitionName);
    return;
  }

  //! \todo Make this really only return vector layers

  names.clear();

  for (int i = 0; i < numIntPartitions(partitionName); i++) {
    string internalName = makeIntPartitionName(partitionName, i);
    File::Partition::Ptr part = partition(internalName);
    if (part)
      part->getLayerNames(names);
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
  //! \todo Make this really only return scalar layers

  names.clear();

  File::Partition::Ptr part = partition(intPartitionName);

  if (!part) {
    Msg::print("getIntScalarLayerNames no partition: " + intPartitionName);
    return;
  }

  part->getLayerNames(names);
}

//----------------------------------------------------------------------------//

void 
Field3DFileBase::getIntVectorLayerNames(vector<string> &names, 
                                        const string &intPartitionName) const
{
  //! \todo Make this really only return vector layers

  names.clear();

  File::Partition::Ptr part = partition(intPartitionName);

  if (!part) {
    Msg::print("getIntVectorLayerNames no partition: " + intPartitionName);    
    return;
  }

  part->getLayerNames(names);
}

//----------------------------------------------------------------------------//

void Field3DFileBase::clear()
{
  if (m_hdf5Base) {
    m_hdf5Base->clear();
    return;
  }

  closeInternal();
  m_partitions.clear();
  m_groupMembership.clear();
}

//----------------------------------------------------------------------------//

bool Field3DFileBase::close()
{
  if (m_hdf5Base) {
    return m_hdf5Base->close();
  }

  closeInternal();

  return true;
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
  if (m_hdf5Base) {
    m_hdf5Base->addGroupMembership(groupMembers);
    return;
  }

  GroupMembershipMap::const_iterator i = groupMembers.begin();
  GroupMembershipMap::const_iterator end = groupMembers.end();

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
  cleanup();
}

//----------------------------------------------------------------------------//

bool Field3DInputFile::open(const string &filename)
{
  clear();

  bool success = true;

  // Record filename
  m_filename = filename;

  try {

    // Throws exceptions if the file doesn't exist.
    checkFile(filename);
    
    // Open the Ogawa archive
    m_archive.reset(new Alembic::Ogawa::IArchive(filename));

    // Error check and HDF5 fallback
    if (!m_archive->isValid()) {
      m_hdf5.reset(new Field3DInputFileHDF5);
      m_hdf5Base = m_hdf5;
      if (m_hdf5->open(filename)) {
        // Handled. Just return.
        return true;
      } else {
        throw NoSuchFileException(filename);
      }
    }

    // Grab the root group
    m_root.reset(new OgIGroup(*m_archive));
    
    // Check version number
    try {
      OgIAttribute<veci32_t> version = 
        m_root->findAttribute<veci32_t>(k_versionAttrName);
      if (!version.isValid()) {
        throw OgIAttributeException("Missing version attribute.");
      }
      int fileVersion[3] = { version.value()[0],
                             version.value()[1],
                             version.value()[2] };
      if (!isSupportedFileVersion(fileVersion, k_minFileVersion)) {
        stringstream versionStr;
        versionStr << fileVersion[0] << "."
                   << fileVersion[1] << "."
                   << fileVersion[2];
        throw UnsupportedVersionException(versionStr.str());
      }
    }
    catch (OgIAttributeException &e) {
      
    }

    // Read the global metadata. This does not always exists, 
    // depends on if it was written or not.
    try { 
      const OgIGroup metadataGroup = m_root->findGroup("field3d_global_metadata");
      if (metadataGroup.isValid()) {
        readMetadata(metadataGroup);
      } 
    }
    catch (...) {
      Msg::print(Msg::SevWarning, 
                 "Unknown error when reading file metadata ");
    }

    // Read the partition and layer info
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
  catch (BadFileHierarchyException &) {
    Msg::print(Msg::SevWarning, 
               "In file: " + filename + " - Bad file hierarchy. ");
    success = false;    
  }
  catch (runtime_error &e) {
    // HDF5 fallback
    m_hdf5.reset(new Field3DInputFileHDF5);
    m_hdf5Base = m_hdf5;
    if (m_hdf5->open(filename)) {
      // Handled. Just return.
      return true;
    } else {
      Msg::print(Msg::SevWarning,
                 "In file: " + filename + ": " + string(e.what()));
      success = false;
    }
  }
  catch (...) {
    Msg::print(Msg::SevWarning, 
               "In file: " + filename + " Unknown exception ");
    success = false;
  }

  if (!success) {
    close();
  }

  return success;
}

//----------------------------------------------------------------------------//

bool Field3DInputFile::readPartitionAndLayerInfo()
{
  // Find all the partition names
  std::vector<std::string> groups = m_root->groupNames();
  
  // Store the partition names
  m_partitions.clear();
  for (std::vector<std::string>::const_iterator i = groups.begin(), 
         end = groups.end(); i != end; ++i) {
    // Grab the name
    const std::string &name = *i;
    // Skip metadata
    if (name == "field3d_global_metadata") {
      continue;
    }
    // Build partition
    File::Partition::Ptr part(new File::Partition);
    part->name = name;
    m_partitions.push_back(part);
  }

  // For each partition, find its mapping ---

  for (PartitionList::iterator i = m_partitions.begin();
       i != m_partitions.end(); ++i) {
    // Grab the name
    const std::string &name = (**i).name;
    // Open the partition group
    const OgIGroup partitionGroup = m_root->findGroup(name);
    if (!partitionGroup.isValid()) {
      Msg::print(Msg::SevWarning, "Couldn't open partition group " + name);
    }
    // Open the mapping group
    const OgIGroup mappingGroup = partitionGroup.findGroup(k_mappingStr);
    if (!mappingGroup.isValid()) {
      Msg::print(Msg::SevWarning, "Couldn't open mapping group " + name);
    }
    // Build the mapping
    FieldMapping::Ptr mapping = readFieldMapping(mappingGroup);
#if 0
    if (!mapping) {
      Msg::print(Msg::SevWarning, "Got a null pointer when reading mapping");
      throw ReadMappingException((**i).name);
    }
#endif
    // Attach the mapping to the partition
    (**i).mapping = mapping;
  }

  // ... And then find its layers ---

  for (PartitionList::const_iterator i = m_partitions.begin();
       i != m_partitions.end(); ++i) {
    // Grab the name
    const std::string &partitionName = (**i).name;
    // Open the partition group
    const OgIGroup partitionGroup = m_root->findGroup(partitionName);
    if (!partitionGroup.isValid()) {
      Msg::print(Msg::SevWarning, "Couldn't open partition group " + 
                 partitionName);
    }
    // Get all the layer names
    groups = partitionGroup.groupNames();
    for (std::vector<std::string>::const_iterator l = groups.begin(), 
           lEnd = groups.end(); l != lEnd; ++l) {
      // Grab layer name
      const std::string layerName = *l;
      // Skip the mapping group
      if (layerName == k_mappingStr) {
        continue;
      }
      // Construct the layer
      File::Layer layer;
      layer.name = *l;
      layer.parent = partitionName;
      // Add to partition
      partition(partitionName)->addLayer(layer);
    }
  }

  return true;
}

//----------------------------------------------------------------------------//

bool Field3DInputFile::readMetadata(const OgIGroup &metadataGroup, 
                                    FieldBase::Ptr field) const
{
  return readMeta(metadataGroup, field->metadata());
}

//----------------------------------------------------------------------------//

bool Field3DInputFile::readMetadata(const OgIGroup &metadataGroup)
{
  return readMeta(metadataGroup, metadata());
}

//----------------------------------------------------------------------------//
// Field3DOutputFile implementations
//----------------------------------------------------------------------------//

bool Field3DOutputFile::ms_doOgawa = true;

//----------------------------------------------------------------------------//

Field3DOutputFile::Field3DOutputFile() 
{ 
  // Empty
}

//----------------------------------------------------------------------------//

Field3DOutputFile::~Field3DOutputFile() 
{ 
  cleanup();
}

//----------------------------------------------------------------------------//

bool Field3DOutputFile::create(const string &filename, CreateMode cm)
{
  if (!ms_doOgawa) {
    m_hdf5.reset(new Field3DOutputFileHDF5);
    m_hdf5Base = m_hdf5;
    int ccm = cm;
    return m_hdf5->create(filename, Field3DOutputFileHDF5::CreateMode(ccm));
  }

  closeInternal();

  if (cm == FailOnExisting && fileExists(filename)) {
    return false;
  }

  // Create the Ogawa archive
  m_archive.reset(new Alembic::Ogawa::OArchive(filename));

  // Check that it's valid
  if (!m_archive->isValid()) {
    m_archive.reset();
    return false;
  }

  // Get the root
  m_root.reset(new OgOGroup(*m_archive));

  // Create the version attribute
  OgOAttribute<veci32_t> f3dVersion(*m_root, k_versionAttrName, 
                                    k_currentFileVersion);

  return true;
}

//----------------------------------------------------------------------------//

bool Field3DOutputFile::writeMapping(OgOGroup &partitionGroup, 
                                     FieldMapping::Ptr mapping)
{
  ClassFactory      &factory   = ClassFactory::singleton();
  const std::string  className = mapping->className();

  try {

    OgOGroup mappingGroup(partitionGroup, k_mappingStr);

    OgOAttribute<string> classNameAttr(mappingGroup, k_mappingTypeAttrName,
                                       className);

    FieldMappingIO::Ptr io = factory.createFieldMappingIO(className);
    if (!io) {
      Msg::print(Msg::SevWarning, "Unable to find class type: " + 
                 className);
      return false;
    }

    return io->write(mappingGroup, mapping);

  }
  catch (OgOGroupException &e) {
    Msg::print(Msg::SevWarning, "Couldn't create group: " + string(e.what()) );
    throw WriteMappingException(k_mappingStr);
  }

}

//----------------------------------------------------------------------------//

bool Field3DOutputFile::writeMetadata(OgOGroup &metadataGroup, 
                                      FieldBase::Ptr field)
{
  {
    FieldMetadata::StrMetadata::const_iterator i = 
      field->metadata().strMetadata().begin();
    FieldMetadata::StrMetadata::const_iterator end = 
      field->metadata().strMetadata().end();
    for (; i != end; ++i) {
      try {
        OgOAttribute<string>(metadataGroup, i->first, i->second);
      }
      catch (OgOAttributeException &e) {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first + 
                   " " + e.what());
        return false;
      }
    }
  }

  {
    FieldMetadata::IntMetadata::const_iterator i = 
      field->metadata().intMetadata().begin();
    FieldMetadata::IntMetadata::const_iterator end = 
      field->metadata().intMetadata().end();
    for (; i != end; ++i) {
      try {
        OgOAttribute<int32_t>(metadataGroup, i->first, i->second);
      }
      catch (OgOAttributeException &e) {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first + 
                   " " + e.what());
        return false;
      }
    }
  }

  {
    FieldMetadata::FloatMetadata::const_iterator i = 
      field->metadata().floatMetadata().begin();
    FieldMetadata::FloatMetadata::const_iterator end = 
      field->metadata().floatMetadata().end();
    for (; i != end; ++i) {
      try {
        OgOAttribute<float32_t>(metadataGroup, i->first, i->second);
      }
      catch (OgOAttributeException &e) {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first + 
                   " " + e.what());
        return false;
      }
    }
  }

  {
    FieldMetadata::VecIntMetadata::const_iterator i = 
      field->metadata().vecIntMetadata().begin();
    FieldMetadata::VecIntMetadata::const_iterator end = 
      field->metadata().vecIntMetadata().end();
    for (; i != end; ++i) {
      try {
        OgOAttribute<veci32_t>(metadataGroup, i->first, i->second);
      }
      catch (OgOAttributeException &e) {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first + 
                   " " + e.what());
        return false;
      }
    }
  }

  {
    FieldMetadata::VecFloatMetadata::const_iterator i = 
      field->metadata().vecFloatMetadata().begin();
    FieldMetadata::VecFloatMetadata::const_iterator end = 
      field->metadata().vecFloatMetadata().end();
    for (; i != end; ++i) {
      try {
        OgOAttribute<vec32_t>(metadataGroup, i->first, i->second);
      }
      catch (OgOAttributeException &e) {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first + 
                   " " + e.what());
        return false;
      }
    }

  }

  return true;

}

//----------------------------------------------------------------------------//

bool Field3DOutputFile::writeMetadata(OgOGroup &metadataGroup)
{
  {
    FieldMetadata::StrMetadata::const_iterator i = 
      metadata().strMetadata().begin();
    FieldMetadata::StrMetadata::const_iterator end = 
      metadata().strMetadata().end();
    for (; i != end; ++i) {
      try {
        OgOAttribute<string>(metadataGroup, i->first, i->second);
      }
      catch (OgOAttributeException &e) {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first + 
                   " " + e.what());
        return false;
      }
    }
  }

  {
    FieldMetadata::IntMetadata::const_iterator i = 
      metadata().intMetadata().begin();
    FieldMetadata::IntMetadata::const_iterator end = 
      metadata().intMetadata().end();
    for (; i != end; ++i) {
      try {
        OgOAttribute<int32_t>(metadataGroup, i->first, i->second);
      }
      catch (OgOAttributeException &e) {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first + 
                   " " + e.what());
        return false;
      }
    }
  }

  {
    FieldMetadata::FloatMetadata::const_iterator i = 
      metadata().floatMetadata().begin();
    FieldMetadata::FloatMetadata::const_iterator end = 
      metadata().floatMetadata().end();
    for (; i != end; ++i) {
      try {
        OgOAttribute<float32_t>(metadataGroup, i->first, i->second);
      }
      catch (OgOAttributeException &e) {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first + 
                   " " + e.what());
        return false;
      }
    }
  }

  {
    FieldMetadata::VecIntMetadata::const_iterator i = 
      metadata().vecIntMetadata().begin();
    FieldMetadata::VecIntMetadata::const_iterator end = 
      metadata().vecIntMetadata().end();
    for (; i != end; ++i) {
      try {
        OgOAttribute<veci32_t>(metadataGroup, i->first, i->second);
      }
      catch (OgOAttributeException &e) {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first + 
                   " " + e.what());
        return false;
      }
    }
  }

  {
    FieldMetadata::VecFloatMetadata::const_iterator i = 
      metadata().vecFloatMetadata().begin();
    FieldMetadata::VecFloatMetadata::const_iterator end = 
      metadata().vecFloatMetadata().end();
    for (; i != end; ++i) {
      try {
        OgOAttribute<vec32_t>(metadataGroup, i->first, i->second);
      }
      catch (OgOAttributeException &e) {
        Msg::print(Msg::SevWarning, "Writing attribute " + i->first + 
                   " " + e.what());
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
  if (m_hdf5) {
    return m_hdf5->writeGlobalMetadata();
  }

  OgOGroup ogMetadata(*m_root, "field3d_global_metadata");
  if (!writeMetadata(ogMetadata)) {
    Msg::print(Msg::SevWarning, "Error writing file metadata.");
    return false;
  } 
 
  return true;
}

//----------------------------------------------------------------------------//

bool 
Field3DOutputFile::writeGroupMembership()
{
  if (m_hdf5) {
    return m_hdf5->writeGroupMembership();
  }

#if 0

  //! \todo Finish

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

#endif
  
  return true;
}

//----------------------------------------------------------------------------//

std::string
Field3DOutputFile::incrementPartitionName(std::string &partitionName)
{
  std::string myPartitionName = removeUniqueId(partitionName);
  int nextIdx = -1;
  if (m_partitionCount.find(myPartitionName) != m_partitionCount.end()) {
    nextIdx = ++m_partitionCount[myPartitionName];
  } else {
    nextIdx = 0;
    m_partitionCount[myPartitionName] = 0;
  }

  return makeIntPartitionName(myPartitionName, nextIdx);
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
    cout << "  Layers: " << endl;
    vector<string> names;
    (**i).getLayerNames(names);
    for_each(names.begin(), names.end(), print<string>(4));    
  }
}

//----------------------------------------------------------------------------//
// Function Implementations
//----------------------------------------------------------------------------//

bool fileExists(const std::string &filename)
{
#ifdef WIN32
  struct __stat64 statbuf;
  return (_stat64(filename.c_str(), &statbuf) != -1);
#else
  struct stat statbuf;
  return (stat(filename.c_str(), &statbuf) != -1);
#endif
}

//----------------------------------------------------------------------------//

File::Partition::Ptr
Field3DOutputFile::createNewPartition(const std::string &partitionName,
                                      const std::string & /* layerName */,
                                      FieldRes::Ptr field)
{
  using namespace Exc;
  
  File::Partition::Ptr newPart(new File::Partition);
  newPart->name = partitionName;

  boost::shared_ptr<OgOGroup> ogPartition(new OgOGroup(*m_root, newPart->name));
  newPart->setGroup(ogPartition);

  m_partitions.push_back(newPart);

  // Pick up new pointer
  File::Partition::Ptr part = partition(partitionName);
  
  // Add mapping group to the partition
  try {
    if (!writeMapping(*ogPartition, field->mapping())) {
      Msg::print(Msg::SevWarning, 
                 "writeMapping returned false for an unknown reason ");
      return File::Partition::Ptr();
    }
  }
  catch (WriteMappingException &e) {
    Msg::print(Msg::SevWarning, "Couldn't write mapping for partition: " 
               + partitionName);
    return File::Partition::Ptr();
  }
  catch (...) {
    Msg::print(Msg::SevWarning, 
               "Unknown error when writing mapping for partition: " 
               + partitionName);
    return File::Partition::Ptr();    
  }

  // Set the mapping of the partition. Since all layers share their 
  // partition's mapping, we can just pick this first one. All subsequent
  // additions to the same partition are checked to have the same mapping
  part->mapping = field->mapping();

  // Tag node as partition
  // Create a version attribute on the root node
  OgOAttribute<string>(*ogPartition, "is_field3d_partition", "1");

  return part;
}

//----------------------------------------------------------------------------//
// Template implementations
//----------------------------------------------------------------------------//

template <class Data_T>
bool Field3DOutputFile::writeLayer(const std::string &userPartitionName, 
                                   const std::string &layerName, 
                                   typename Field<Data_T>::Ptr field)
{
  using std::string;

  // Null pointer check
  if (!field) {
    Msg::print(Msg::SevWarning,
               "Called writeLayer with null pointer. Ignoring...");
    return false;
  }
  
  // Make sure archive is open
  if (!m_archive) {
    Msg::print(Msg::SevWarning, 
               "Attempting to write layer without opening file first.");
    return false;
  }

  // Get the partition name
  string partitionName = intPartitionName(userPartitionName, layerName, field);

  // Get the partition
  File::Partition::Ptr part = partition(partitionName);

  if (!part) {
    // Create a new partition
    part = createNewPartition(partitionName, layerName, field);
    // Make sure it was created 
    if (!part) {
      return false;
    }
  } else {
    // Check that we have a valid mapping
    if (!field->mapping()) {
      Msg::print(Msg::SevWarning, 
                 "Couldn't add layer \"" + layerName + "\" to partition \""
                 + partitionName + "\" because the layer's mapping is null.");
      return false;    
    }
    // Check if the layer already exists. If it does, we need to make a 
    // different partition
    if (part->layer(layerName)) {
      // Increment the internal partition name
      partitionName = incrementPartitionName(partitionName);
      // Create a new partition
      part = createNewPartition(partitionName, layerName, field);
      // Make sure it was created 
      if (!part) {
        return false;
      }
    }
  }

  // Check mapping not null
  if (!part->mapping) {
    Msg::print(Msg::SevWarning, "Severe error - partition mapping is null: " 
              + partitionName);
    return false;    
  }

  // Check that the mapping matches what's already in the Partition
  if (!field->mapping()->isIdentical(part->mapping)) {
    Msg::print(Msg::SevWarning, "Couldn't add layer \"" + layerName 
              + "\" to partition \"" + partitionName 
              + "\" because mapping doesn't match");
    return false;
  }

  // Open the partition

  OgOGroup &ogPartition = part->group();

  // Build a Layer

  File::Layer layer;
  layer.name   = layerName;
  layer.parent = partitionName;

  // Add Layer to file ---

  OgOGroup ogLayer(ogPartition, layerName);

  // Tag as layer
  OgOAttribute<string> classType(ogLayer, "class_type", "field3d_layer");

  // Create metadata
  OgOGroup ogMetadata(ogLayer, "metadata");

  // Write metadata
  writeMetadata(ogMetadata, field);

  // Write field data
  writeField(ogLayer, field);

  // Add to partition

  part->addLayer(layer);

  return true;
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename Field<Data_T>::Ptr
Field3DInputFile::readLayer(const std::string &intPartitionName,
                            const std::string &layerName) const
{
  typedef typename Field<Data_T>::Ptr FieldPtr;

  // Instantiate a null pointer for easier code reading
  FieldPtr nullPtr;

  // Find the partition
  File::Partition::Ptr part = partition(intPartitionName);
  if (!part) {
    Msg::print(Msg::SevWarning, "Couldn't find partition: " + intPartitionName);
    return nullPtr;
  }

  // Find the layer
  const File::Layer *layer = part->layer(layerName);
  if (!layer) {
    Msg::print(Msg::SevWarning, "Couldn't find layer: " + layerName);
    return nullPtr;
  }

  // Open the partition group
  const OgIGroup partitionGroup = m_root->findGroup(intPartitionName);
  if (!partitionGroup.isValid()) {
    Msg::print(Msg::SevWarning, "Couldn't open partition group " + 
               intPartitionName);
    return nullPtr;
  }

  // Open the layer group
  const OgIGroup layerGroup = partitionGroup.findGroup(layerName);
  if (!layerGroup.isValid()) {
    Msg::print(Msg::SevWarning, "Couldn't open layer group " + 
               layerName);
    return nullPtr;
  }

  // Get the class name
  string layerPath = layer->parent + "/" + layer->name;
  string className;
  try {
    className = layerGroup.findAttribute<string>("class_name").value();
  }
  catch (OgIAttributeException &e) {
    Msg::print(Msg::SevWarning, "Couldn't find class_name attrib in layer " + 
               layerName);
    return nullPtr;
  }
  
  // Check the cache

  FieldCache<Data_T> &cache       = FieldCache<Data_T>::singleton();
  FieldPtr            cachedField = cache.getCachedField(m_filename, layerPath);

  if (cachedField) {
    return cachedField;
  } 

  // Construct the field and load the data

  typename Field<Data_T>::Ptr field;
  field = readField<Data_T>(className, layerGroup, m_filename, layerPath);

  if (!field) {
    // This isn't really an error
    return nullPtr;
  }
  
  // Read the metadata
  const OgIGroup metadataGroup = layerGroup.findGroup("metadata");
  if (metadataGroup.isValid()) {
    readMetadata(metadataGroup, field);
  }
  
  // Set the name of the field appropriately
  field->name      = removeUniqueId(intPartitionName);
  field->attribute = layerName;
  field->setMapping(part->mapping);

  // Cache the field for future use
  if (field) {
    cache.cacheField(field, m_filename, layerPath);
  }

  return field;
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename Field<Data_T>::Vec
Field3DInputFile::readLayers(const std::string &name) const
{
  using std::vector;
  using std::string;

  typedef typename Field<Data_T>::Ptr FieldPtr;
  typedef typename Field<Data_T>::Vec FieldList;
  
  FieldList ret;
  std::vector<std::string> parts;
  getIntPartitionNames(parts);

  for (vector<string>::iterator p = parts.begin(); p != parts.end(); ++p) {
    vector<std::string> layers;
    getIntScalarLayerNames(layers, *p);
    for (vector<string>::iterator l = layers.begin(); l != layers.end(); ++l) {
      // Only read if it matches the name
      if ((name.length() == 0) || (*l == name)) {
        FieldPtr mf = readLayer<Data_T>(*p, *l);
        if (mf) {
          ret.push_back(mf);
        }
      }
    }
  }
  
  return ret;
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename Field<Data_T>::Vec
Field3DInputFile::readLayers(const std::string &partitionName, 
                             const std::string &layerName) const
{
  using namespace std;
  
  typedef typename Field<Data_T>::Ptr FieldPtr;
  typedef typename Field<Data_T>::Vec FieldList;

  FieldList ret;

  if ((layerName.length() == 0) || (partitionName.length() == 0))
    return ret;
  
  std::vector<std::string> parts;
  getIntPartitionNames(parts);
 
  for (vector<string>::iterator p = parts.begin(); p != parts.end(); ++p) {
    std::vector<std::string> layers;
    getIntScalarLayerNames(layers, *p);
    if (removeUniqueId(*p) == partitionName) {
      for (vector<string>::iterator l = layers.begin(); 
           l != layers.end(); ++l) {
        // Only read if it matches the name
        if (*l == layerName) {
          FieldPtr mf = readLayer<Data_T>(*p, *l);
          if (mf)
            ret.push_back(mf);
        }
      }
    }
  }
  
  return ret;
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename EmptyField<Data_T>::Ptr 
Field3DInputFile::readProxyLayer(OgIGroup &location, 
                                 const std::string &name,
                                 const std::string &attribute,
                                 FieldMapping::Ptr mapping) const
{
  using namespace boost;
  using namespace std;

  typename EmptyField<Data_T>::Ptr null;

  const std::string extentsMinStr("extents_min");
  const std::string extentsMaxStr("extents_max");
  const std::string dataWindowMinStr("data_window_min");
  const std::string dataWindowMaxStr("data_window_max");

  Box3i extents, dataW;
  
  // Get extents ---

  OgIAttribute<veci32_t> extMinAttr = 
    location.findAttribute<veci32_t>(extentsMinStr);
  OgIAttribute<veci32_t> extMaxAttr = 
    location.findAttribute<veci32_t>(extentsMaxStr);
  if (!extMinAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    extentsMinStr);
  }
  if (!extMaxAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    extentsMaxStr);
  }

  extents.min = extMinAttr.value();
  extents.max = extMaxAttr.value();

  // Get data window ---

  OgIAttribute<veci32_t> dwMinAttr = 
    location.findAttribute<veci32_t>(dataWindowMinStr);
  OgIAttribute<veci32_t> dwMaxAttr = 
    location.findAttribute<veci32_t>(dataWindowMaxStr);
  if (!dwMinAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    dataWindowMinStr);
  }
  if (!dwMaxAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    dataWindowMaxStr);
  }

  dataW.min = dwMinAttr.value();
  dataW.max = dwMaxAttr.value();

  // Construct the field
  typename EmptyField<Data_T>::Ptr field(new EmptyField<Data_T>);
  field->setSize(extents, dataW);

  // Read the metadata 
  OgIGroup metadataGroup = location.findGroup("metadata");
  if (metadataGroup.isValid()) {    
    readMetadata(metadataGroup, field);
  }

  // Set field properties
  field->name = name;
  field->attribute = attribute;
  field->setMapping(mapping);

  return field;
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename EmptyField<Data_T>::Vec
Field3DInputFile::readProxyLayer(const std::string &partitionName, 
                                 const std::string &layerName,
                                 bool isVectorLayer) const
{
  using namespace boost;
  using namespace std;
  using namespace Hdf5Util;

  if (m_hdf5) {
    return m_hdf5->readProxyLayer<Data_T>(partitionName, layerName, 
                                          isVectorLayer);
  }

  // Instantiate a null pointer for easier code reading
  typename EmptyField<Data_T>::Vec emptyList, output;

  if ((layerName.length() == 0) || (partitionName.length() == 0))
    return emptyList;

  std::vector<std::string> parts, layers;
  getIntPartitionNames(parts);
 
  bool foundPartition = false;

  for (vector<string>::iterator p = parts.begin(); p != parts.end(); ++p) {
    if (removeUniqueId(*p) == partitionName) {
      foundPartition = true;
      if (isVectorLayer) {
        getIntVectorLayerNames(layers, *p);
      } else {
        getIntScalarLayerNames(layers, *p);
      }
      for (vector<string>::iterator l = layers.begin(); 
           l != layers.end(); ++l) {
        if (*l == layerName) {
          // Find the partition
          File::Partition::Ptr part = partition(*p);
          if (!part) {
            Msg::print(Msg::SevWarning, "Couldn't find partition: " + *p);
            return emptyList;
          }
          // Find the layer
          const File::Layer *layer;
          if (isVectorLayer) {
            layer = part->layer(layerName);
          } else {
            layer = part->layer(layerName);
          }
          if (!layer) {
            Msg::print(Msg::SevWarning, "Couldn't find layer: " + layerName);
            return emptyList;
          }
          // Open the layer group
          string layerPath = layer->parent + "/" + layer->name;
          OgIGroup parent = m_root->findGroup(layer->parent);
          if (!parent.isValid()) {
            Msg::print(Msg::SevWarning, "Couldn't find layer parent " 
                      + layerPath + " in .f3d file ");
            return emptyList;
          }
          OgIGroup layerGroup = parent.findGroup(layer->name);
          if (!layerGroup.isValid()) {
            Msg::print(Msg::SevWarning, "Couldn't find layer group " 
                      + layerPath + " in .f3d file ");
            return emptyList;
          }

          // Make the proxy representation
          typename EmptyField<Data_T>::Ptr field = 
            readProxyLayer<Data_T>(layerGroup, partitionName, layerName, 
                                   part->mapping);

          // Read MIPField's number of mip levels
          int numLevels = 0;
          OgIGroup mipGroup = layerGroup.findGroup("mip_levels");
          if (mipGroup.isValid()) {
            OgIAttribute<uint32_t> levelsAttr = 
              mipGroup.findAttribute<uint32_t>("levels");
            if (levelsAttr.isValid()) {
              numLevels = levelsAttr.value();
            }
          }
          field->metadata().setIntMetadata("mip_levels", numLevels);

          // Add field to output
          output.push_back(field);
        }
      }
    }
  }

  if (!foundPartition) {
    Msg::print(Msg::SevWarning, "Couldn't find partition: " + partitionName);
    return emptyList;    
  }
  
  return output;
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename EmptyField<Data_T>::Vec
Field3DInputFile::readProxyScalarLayers(const std::string &name) const
{
  using namespace std;

  typedef typename EmptyField<Data_T>::Ptr FieldPtr;
  typedef std::vector<FieldPtr> FieldList;
  
  FieldList ret;
  
  std::vector<std::string> parts;
  getPartitionNames(parts);
  
  for (vector<string>::iterator p = parts.begin(); p != parts.end(); ++p) {
  std::vector<std::string> layers;
    getScalarLayerNames(layers, *p);
    for (vector<string>::iterator l = layers.begin(); l != layers.end(); ++l) {
      // Only read if it matches the name
      if ((name.length() == 0) || (*l == name)) {
        FieldList f = readProxyLayer<Data_T>(*p, *l, false);
        for (typename FieldList::iterator i = f.begin(); i != f.end(); ++i) {
          if (*i) {
            ret.push_back(*i);
          }
        }
      }
    }
  }
  
  return ret;
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename EmptyField<Data_T>::Vec
Field3DInputFile::readProxyVectorLayers(const std::string &name) const
{
  using namespace std;
  
  typedef typename EmptyField<Data_T>::Ptr FieldPtr;
  typedef std::vector<FieldPtr> FieldList;
  
  FieldList ret;
  
  std::vector<std::string> parts;
  getPartitionNames(parts);
  
  for (vector<string>::iterator p = parts.begin(); p != parts.end(); ++p) {
  std::vector<std::string> layers;
    getVectorLayerNames(layers, *p);
    for (vector<string>::iterator l = layers.begin(); l != layers.end(); ++l) {
      // Only read if it matches the name
      if ((name.length() == 0) || (*l == name)) {
        FieldList f = readProxyLayer<Data_T>(*p, *l, true);
        for (typename FieldList::iterator i = f.begin(); i != f.end(); ++i) {
          if (*i) {
            ret.push_back(*i);
          }
        }
      }
    }
  }
  
  return ret;  
}

//----------------------------------------------------------------------------//
// Template instantiations
//----------------------------------------------------------------------------//

#define FIELD3D_INSTANTIATION_WRITELAYER(type)                          \
  template                                                              \
  bool Field3DOutputFile::writeLayer<type>                              \
  (const std::string &, const std::string &, Field<type>::Ptr );        \
  
FIELD3D_INSTANTIATION_WRITELAYER(float16_t);
FIELD3D_INSTANTIATION_WRITELAYER(float32_t);
FIELD3D_INSTANTIATION_WRITELAYER(float64_t);
FIELD3D_INSTANTIATION_WRITELAYER(vec16_t);
FIELD3D_INSTANTIATION_WRITELAYER(vec32_t);
FIELD3D_INSTANTIATION_WRITELAYER(vec64_t);

//----------------------------------------------------------------------------//

#if 0

#define FIELD3D_INSTANTIATION_READLAYER(type)                           \
  template                                                              \
  Field<type>::Ptr                                                      \
  Field3DInputFile::readLayer<type>                                     \
  (const std::string &, const std::string &) const;                     \
  
FIELD3D_INSTANTIATION_READLAYER(float16_t);
FIELD3D_INSTANTIATION_READLAYER(float32_t);
FIELD3D_INSTANTIATION_READLAYER(float64_t);
FIELD3D_INSTANTIATION_READLAYER(vec16_t);
FIELD3D_INSTANTIATION_READLAYER(vec32_t);
FIELD3D_INSTANTIATION_READLAYER(vec64_t);

#endif

//----------------------------------------------------------------------------//

#define FIELD3D_INSTANTIATION_READLAYERS1(type)                         \
  template                                                              \
  Field<type>::Vec                                                      \
  Field3DInputFile::readLayers<type>(const std::string &name) const;    \

FIELD3D_INSTANTIATION_READLAYERS1(float16_t);
FIELD3D_INSTANTIATION_READLAYERS1(float32_t);
FIELD3D_INSTANTIATION_READLAYERS1(float64_t);
FIELD3D_INSTANTIATION_READLAYERS1(vec16_t);
FIELD3D_INSTANTIATION_READLAYERS1(vec32_t);
FIELD3D_INSTANTIATION_READLAYERS1(vec64_t);

//----------------------------------------------------------------------------//

#define FIELD3D_INSTANTIATION_READLAYERS2(type)                         \
  template                                                              \
  Field<type>::Vec                                                      \
  Field3DInputFile::readLayers<type>(const std::string &partitionName,    \
                                     const std::string &layerName) const; \

FIELD3D_INSTANTIATION_READLAYERS2(float16_t);
FIELD3D_INSTANTIATION_READLAYERS2(float32_t);
FIELD3D_INSTANTIATION_READLAYERS2(float64_t);
FIELD3D_INSTANTIATION_READLAYERS2(vec16_t);
FIELD3D_INSTANTIATION_READLAYERS2(vec32_t);
FIELD3D_INSTANTIATION_READLAYERS2(vec64_t);

//----------------------------------------------------------------------------//

#define FIELD3D_INSTANTIATION_READPROXYLAYER(type)                      \
  template                                                              \
  EmptyField<type>::Vec                                                 \
  Field3DInputFile::readProxyLayer<type>(const std::string &partitionName, \
                                         const std::string &layerName,  \
                                         bool isVectorLayer) const      \
  
FIELD3D_INSTANTIATION_READPROXYLAYER(float16_t);
FIELD3D_INSTANTIATION_READPROXYLAYER(float32_t);
FIELD3D_INSTANTIATION_READPROXYLAYER(float64_t);
FIELD3D_INSTANTIATION_READPROXYLAYER(vec16_t);
FIELD3D_INSTANTIATION_READPROXYLAYER(vec32_t);
FIELD3D_INSTANTIATION_READPROXYLAYER(vec64_t);

//----------------------------------------------------------------------------//

#define FIELD3D_INSTANTIATION_READPROXYSCALARLAYER(type)                \
  template                                                              \
  EmptyField<type>::Vec                                                 \
  Field3DInputFile::readProxyScalarLayers<type>                         \
  (const std::string &name) const                                       \
  
FIELD3D_INSTANTIATION_READPROXYSCALARLAYER(float16_t);
FIELD3D_INSTANTIATION_READPROXYSCALARLAYER(float32_t);
FIELD3D_INSTANTIATION_READPROXYSCALARLAYER(float64_t);
FIELD3D_INSTANTIATION_READPROXYSCALARLAYER(vec16_t);
FIELD3D_INSTANTIATION_READPROXYSCALARLAYER(vec32_t);
FIELD3D_INSTANTIATION_READPROXYSCALARLAYER(vec64_t);

//----------------------------------------------------------------------------//

#define FIELD3D_INSTANTIATION_READPROXYVECTORLAYER(type)                \
  template                                                              \
  EmptyField<type>::Vec                                                 \
  Field3DInputFile::readProxyVectorLayers<type>                         \
  (const std::string &name) const                                       \
  
FIELD3D_INSTANTIATION_READPROXYVECTORLAYER(float16_t);
FIELD3D_INSTANTIATION_READPROXYVECTORLAYER(float32_t);
FIELD3D_INSTANTIATION_READPROXYVECTORLAYER(float64_t);
FIELD3D_INSTANTIATION_READPROXYVECTORLAYER(vec16_t);
FIELD3D_INSTANTIATION_READPROXYVECTORLAYER(vec32_t);
FIELD3D_INSTANTIATION_READPROXYVECTORLAYER(vec64_t);

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
