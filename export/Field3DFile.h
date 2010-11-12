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

/*! \file Field3DFile.h
  \brief Contains the Field3DFile classes
  \ingroup field

  OSS sanitized
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_Field3DFile_H_
#define _INCLUDED_Field3D_Field3DFile_H_

//----------------------------------------------------------------------------//

#include <list>
#include <string>
#include <vector>

#include <hdf5.h>

#include <boost/intrusive_ptr.hpp>

#include "EmptyField.h"
#include "Field.h"
#include "FieldMetadata.h"
#include "ClassFactory.h"
#include "Hdf5Util.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN



//----------------------------------------------------------------------------//
// Function Declarations
//----------------------------------------------------------------------------//

//! \name classFactory IO functions
// \{

//! This function creates a FieldIO instance based on className
//! which then reads the field data from layerGroup location
template <class Data_T>
typename Field<Data_T>::Ptr 
readField(const std::string &className, hid_t layerGroup,
          const std::string &filename, const std::string &layerPath);
//! This function creates a FieldIO instance based on field->className()
//! which then writes the field data in layerGroup location
bool writeField(hid_t layerGroup, FieldBase::Ptr field);

//! This function creates a FieldMappingIO instance based on className 
//! read from mappingGroup location which then reads FieldMapping data
FieldMapping::Ptr readFieldMapping(hid_t mappingGroup);

//! This function creates a FieldMappingIO instance based on
//! mapping->className() which then writes FieldMapping 
//! data to mappingGroup location
bool writeFieldMapping(hid_t mappingGroup, FieldMapping::Ptr mapping);

//! \}

//----------------------------------------------------------------------------//
// Layer
//----------------------------------------------------------------------------//

//! Namespace for file I/O specifics
//! \ingroup file_int
namespace File {

/*! \class Layer
  \ingroup file_int
  This class wraps up information about a single "Layer" in a .f3d file.
  A layer is a "Field" with a name. The mapping lives on the Partition object,
  so the layer only knows about the location of the field in the file.
*/

class Layer
{
public:
  //! The name of the layer (always available)
  std::string name;
  //! The name of the parent partition. We need this in order to open
  //! its group.
  std::string parent;
};
  
} // namespace File

//----------------------------------------------------------------------------//
// Partition
//----------------------------------------------------------------------------//

namespace File {

/*! \class Partition
  \ingroup file_int
  This class represents the partition-level node in a f3D file. 
  The partition contains one "Mapping" and N "Fields" that all share that
  mapping.
*/

class Partition : public RefBase
{
public:

  typedef std::vector<Layer> ScalarLayerList;
  typedef std::vector<Layer> VectorLayerList;

  typedef boost::intrusive_ptr<Partition> Ptr;
  typedef boost::intrusive_ptr<const Partition> CPtr;

  // Ctors, dtor ---------------------------------------------------------------

  //! Ctor
  Partition() : RefBase() {}

  // Main methods --------------------------------------------------------------

  //! Adds a scalar layer
  void addScalarLayer(const File::Layer &layer);
  //! Adds a vector layer
  void addVectorLayer(const File::Layer &layer);

  //! Finds a scalar layer
  const File::Layer* scalarLayer(const std::string &name) const;
  //! Finds a vector layer
  const File::Layer* vectorLayer(const std::string &name) const;
  
  //! Gets all the scalar layer names. 
  void getScalarLayerNames(std::vector<std::string> &names) const;
  //! Gets all the vector layer names
  void getVectorLayerNames(std::vector<std::string> &names) const;

  // Public data members -------------------------------------------------------

  //! Name of the partition
  std::string name;
  //! Pointer to the mapping object.
  FieldMapping::Ptr mapping;

private:

  // Private data members ------------------------------------------------------

  //! The scalar-valued layers belonging to this partition
  ScalarLayerList m_scalarLayers;
  //! The vector-valued layers belonging to this partition
  VectorLayerList m_vectorLayers;

};


} // namespace File

//----------------------------------------------------------------------------//
// Field3DFileBase
//----------------------------------------------------------------------------//

/*! \class Field3DFileBase
  \ingroup file
  Provides some common functionality for Field3DInputFile and
  Field3DOutputFile. It hold the partition->layer data structures, but
  knows nothing about how to actually get them to/from disk. 
*/

//----------------------------------------------------------------------------//

class Field3DFileBase
{
public:

  // Structs -------------------------------------------------------------------

  struct LayerInfo 
  {
    std::string name;
    std::string parentName;
    int components;  
    LayerInfo(std::string par, std::string nm, int cpt) 
      : name(nm), parentName(par), components(cpt) 
    { /* Empty */ }
  };

  // Typedefs ------------------------------------------------------------------
 
  typedef std::map<std::string, std::string> GroupMembershipMap;

  // Ctor, dtor ----------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  Field3DFileBase();
  //! Pure virtual destructor to ensure we never instantiate this class
  virtual ~Field3DFileBase() = 0;

  //! \}

  // Main methods --------------------------------------------------------------

  //! Clear the data structures and close the file.
  void clear();

  //! Closes the file. No need to call this unless you specifically want to
  //! close the file early. It will close once the File object goes out of 
  //! scope.
  bool close();

  //! \name Retreiving partition and layer names
  //! \{

  //! Gets the names of all the partitions in the file
  void getPartitionNames(std::vector<std::string> &names) const;
  //! Gets the names of all the scalar layers in a given partition
  void getScalarLayerNames(std::vector<std::string> &names, 
                           const std::string &partitionName) const;
  //! Gets the names of all the vector layers in a given partition
  void getVectorLayerNames(std::vector<std::string> &names, 
                           const std::string &partitionName) const;

  //! Returns a pointer to the given partition
  //! \returns NULL if no partition was found of that name
  File::Partition::Ptr getPartition(const std::string &partitionName) const
  { return partition(partitionName); }

  //! \}

  //! \name Convenience methods for partitionName
  //! \{

  //! Returns a unique partition name given the requested name. This ensures
  //! that partitions with matching mappings get the same name but each
  //! subsequent differing mapping gets a new, separate name
  std::string intPartitionName(const std::string &partitionName,
                               const std::string &layerName,
                               FieldRes::Ptr field);

  //! Strips any unique identifiers from the partition name and returns
  //! the original name
  std::string removeUniqueId(const std::string &partitionName) const;

  //! Add to the group membership
  void addGroupMembership(const GroupMembershipMap &groupMembers);

  //! \}

  // Access to metadata --------------------------------------------------------  

  //! accessor to the m_metadata class
  FieldMetadata<Field3DFileBase>& metadata()
  { return m_metadata; }

  //! Read only access to the m_metadata class
  const FieldMetadata<Field3DFileBase>& metadata() const
  { return m_metadata; }
 
  //! This function should implemented by concrete classes to  
  //! get the callback when metadata changes
  virtual void metadataHasChanged(const std::string &/* name */) 
  { /* Empty */ }

  // Debug ---------------------------------------------------------------------

  //! \name Debug
  //! \{

  void printHierarchy() const;

  //! \}

protected:

  // Internal typedefs ---------------------------------------------------------

  typedef std::vector<File::Partition::Ptr> PartitionList;
  typedef std::map<std::string, int> PartitionCountMap;

  // Convenience methods -------------------------------------------------------

  //! \name Convenience methods
  //! \{

  //! Closes the file if open.
  void closeInternal();
  //! Returns a pointer to the given partition
  //! \returns NULL if no partition was found of that name
  File::Partition::Ptr partition(const std::string &partitionName);
  //! Returns a pointer to the given partition
  //! \returns NULL if no partition was found of that name
  File::Partition::Ptr partition(const std::string &partitionName) const;
  
  //! Gets the names of all the -internal- partitions in the file
  void getIntPartitionNames(std::vector<std::string> &names) const;
  //! Gets the names of all the scalar layers in a given partition, but
  //! assumes that partition name is the -internal- partition name
  void getIntScalarLayerNames(std::vector<std::string> &names, 
                              const std::string &intPartitionName) const;
  //! Gets the names of all the vector layers in a given partition, but
  //! assumes that partition name is the -internal- partition name
  void getIntVectorLayerNames(std::vector<std::string> &names, 
                              const std::string &intPartitionName) const;
  
  //! Returns the number of internal partitions for a given partition name
  int numIntPartitions(const std::string &partitionName) const;

  //! Makes an internal partition name given the external partition name.
  //! Effectively just tacks on .X to the name, where X is the number
  std::string makeIntPartitionName(const std::string &partitionsName,
                                   int i) const;

  //! \}

  // Data members --------------------------------------------------------------

  //! This stores layer info
  std::vector<LayerInfo> m_layerInfo;

  //! The hdf5 id of the current file. Will be -1 if no file is open.
  hid_t m_file;
  //! Vector of partitions. 
  PartitionList m_partitions;
  //! This stores partition names
  std::vector<std::string> m_partitionNames;

  //! Contains a counter for each partition name. This is used to keep multiple
  //! fields with the same name unique in the file
  PartitionCountMap m_partitionCount;

  //! Keeps track of group membership for each layer of partition name.
  //! The key is the "group" and the value is a space separated list of 
  //! "partitionName.0:Layer1 partitionName.1:Layer0  ..."  
  GroupMembershipMap m_groupMembership;

  //! metadata
  FieldMetadata<Field3DFileBase> m_metadata;

private:

  // Private member functions --------------------------------------------------

  Field3DFileBase(const Field3DFileBase&);
  void operator =(const Field3DFileBase&); 


};

//----------------------------------------------------------------------------//
// Field3DInputFile
//----------------------------------------------------------------------------//

/*! \class Field3DInputFile
  \brief Provides reading of .f3d (internally, hdf5) files.
  \ingroup file

  Refer to \ref using_files for examples of how to use this in your code.

  \note We distinguish between scalar and vector layers even though both
  are templated. A scalarField<float> layer is interchangeable with a 
  scalarField<double> (conceptually) but not with a scalar<V3f>, 
  and thus not with vectorField<float>.

 */

//----------------------------------------------------------------------------//

class Field3DInputFile : public Field3DFileBase 
{
public:

  // Ctors, dtor ---------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  Field3DInputFile();
  virtual ~Field3DInputFile();

  //! \}

  // Main interface ------------------------------------------------------------

  //! \name Reading layers from disk
  //! \{

  //! Retrieves all the layers of scalar type and maintains their on-disk
  //! data types
  //! \param layerName If a string is passed in, only layers of that name will
  //! be read from disk.
  template <class Data_T>
  typename Field<Data_T>::Vec
  readScalarLayers(const std::string &layerName = std::string("")) const;

  //! This one allows the allows the partitionName to be passed in
  template <class Data_T>
  typename Field<Data_T>::Vec
  readScalarLayers(const std::string &partitionName, 
                   const std::string &layerName) const;

  //! Retrieves all the layers of vector type and maintains their on-disk
  //! data types
  //! \param layerName If a string is passed in, only layers of that name will
  //! be read from disk.
  template <class Data_T>
  typename Field<FIELD3D_VEC3_T<Data_T> >::Vec
  readVectorLayers(const std::string &layerName = std::string("")) const;

  //! This version allows you to pass in the partition name
  template <class Data_T>
  typename Field<FIELD3D_VEC3_T<Data_T> >::Vec
  readVectorLayers(const std::string &partitionName, 
                   const std::string &layerName) const;

  //! Retrieves all layers for all partitions.
  //! Converts it to the given template type if needed
  template <template <typename T> class Field_T, class Data_T>
  typename Field_T<Data_T>::Vec
  readScalarLayersAs(const std::string &layerName = std::string("")) const;

  //! Retrieves a layers given their and its parent partition's name.
  //! Converts it to the given template type if needed
  template <template <typename T> class Field_T, class Data_T>
  typename Field_T<Data_T>::Vec
  readScalarLayersAs(const std::string &partitionName, 
                     const std::string &layerName) const;

  //! Retrieves a layers for all partitions.
  //! Converts it to the given template type if needed
  template <template <typename T> class Field_T, class Data_T>
  typename Field_T<Data_T>::Vec
  readVectorLayersAs(const std::string &layerName = std::string("")) const;

  //! Retrieves a layers given their and its parent partition's name.
  //! Converts it to the given template type if needed
  template <template <typename T> class Field_T, class Data_T>
  typename Field_T<Data_T>::Vec
  readVectorLayersAs(const std::string &partitionName, 
                     const std::string &layerName) const;

  //! \}

  //! \name Reading proxy data from disk
  //! \{

  //! Retrieves a proxy version (EmptyField) of each layer 
  //! \param name If a string is passed in, only layers of that name will
  //! be read from disk.
  template <class Data_T>
  typename EmptyField<Data_T>::Vec
  readProxyLayer(const std::string &partitionName, 
                 const std::string &layerName, 
                 bool isVectorLayer) const;

  //! Retrieves a proxy version (EmptyField) of each scalar layer 
  //! \param name If a string is passed in, only layers of that name will
  //! be read from disk.
  template <class Data_T>
  typename EmptyField<Data_T>::Vec
  readProxyScalarLayers(const std::string &name = std::string("")) const;

  //! Retrieves a proxy version (EmptyField) of each vector layer 
  //! \param name If a string is passed in, only layers of that name will
  //! be read from disk.
  template <class Data_T>
  typename EmptyField<Data_T>::Vec
  readProxyVectorLayers(const std::string &name = std::string("")) const;

  //! \}

  // File IO ---

  //! Opens the given file
  //! \returns Whether successful
  bool open(const std::string &filename);

  // Callback convenience methods ----------------------------------------------

  //! \name Internal utility methods 
  //! \{

  //! Gets called from parsePartitions. Not intended for any other use.
  herr_t parsePartition(hid_t loc_id, const std::string partitionName);

  //! Gets called from parsePartitions. Not intended for any other use.
  herr_t parseLayer(hid_t loc_id, const std::string &partitionName,
                    const std::string &layerName);

  //! \}

  // Convenience methods -------------------------------------------------------

  //! Read the group membership for the partitions
  bool readGroupMembership(GroupMembershipMap &gpMembershipMap);


private:

  // Convenience methods -------------------------------------------------------

  //! Retrieves a single layer given its and its parent partition's name.
  //! Maintains the on-disk data types
  template <class Data_T>
  typename Field<Data_T>::Ptr
  readScalarLayer(const std::string &intPartitionName, 
                  const std::string &layerName) const;

  //! Retrieves a single layer given its and its parent partition's name.
  //! Maintains the on-disk data types
  template <class Data_T>
  typename Field<FIELD3D_VEC3_T<Data_T> >::Ptr
  readVectorLayer(const std::string &intPartitionName, 
                  const std::string &layerName) const;
  
  //! This call does the actual reading of a layer. Notice that it expects
  //! a unique -internal- partition name.
  template <class Data_T>
  typename Field<Data_T>::Ptr 
  readLayer(const std::string &intPartitionName, 
            const std::string &layerName,
            bool isVectorLayer) const;

  //! Sets up all the partitions and layers, but does not load any data
  bool readPartitionAndLayerInfo();

  //! Read metadata for this layer
  bool readMetadata(hid_t metadata_id, FieldBase::Ptr field) const;

  //! Read global metadata for this file
  bool readMetadata(hid_t metadata_id);

  // Data members --------------------------------------------------------------

  //! Filename, only to be set by open().
  std::string m_filename;  

};

//----------------------------------------------------------------------------//
// Field3DOutputFile
//----------------------------------------------------------------------------//

/*! \class Field3DOutputFile
  \ingroup file
  \brief Provides writing of .f3d (internally, hdf5) files.

  Refer to \ref using_files for examples of how to use this in your code.

  \note We distinguish between scalar and vector layers even though both
  are templated. A scalarField<float> layer is interchangeable with a 
  scalarField<double> (conceptually) but not with a scalar<V3f>, 
  and thus not with vectorField<float>.

 */

//----------------------------------------------------------------------------//

class Field3DOutputFile : public Field3DFileBase 
{
public:

  // Enums ---------------------------------------------------------------------

  enum CreateMode {
    OverwriteMode,
    FailOnExisting
  };

  // Ctors, dtor ---------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  Field3DOutputFile();
  virtual ~Field3DOutputFile();

  //! \}

  // Main interface ------------------------------------------------------------

  //! \name Writing layer to disk
  //! \{

  //! Writes a scalar layer to the "Default" partition.
  template <class Data_T>
  bool writeScalarLayer(const std::string &layerName, 
                        typename Field<Data_T>::Ptr layer)
  { return writeScalarLayer<Data_T>(layerName, std::string("default"), layer); }

  //! Writes a vector layer to the "Default" partition.
  template <class Data_T>
  bool writeVectorLayer(const std::string &layerName, 
                        typename Field<FIELD3D_VEC3_T<Data_T> >::Ptr layer)
  { return writeVectorLayer<Data_T>(layerName, std::string("default"), layer); }

  //! Writes a layer to a specific partition. The partition will be created if
  //! not specified.
  template <class Data_T>
  bool writeScalarLayer(const std::string &partitionName, 
                        const std::string &layerName, 
                        typename Field<Data_T>::Ptr layer);

  //! Writes a layer to a specific partition. The field name and attribute
  //! name are used for partition and layer, respectively
  template <class Data_T>
  bool writeScalarLayer(typename Field<Data_T>::Ptr layer);

  //! Writes a layer to a specific partition. The partition will be created if
  //! not specified.
  template <class Data_T>
  bool writeVectorLayer(const std::string &partitionName, 
                        const std::string &layerName, 
                        typename Field<FIELD3D_VEC3_T<Data_T> >::Ptr layer);

  //! Writes a layer to a specific partition. The field name and attribute
  //! name are used for partition and layer, respectively
  template <class Data_T>
  bool writeVectorLayer(typename Field<FIELD3D_VEC3_T<Data_T> >::Ptr layer);

  //! \}

  //! Creates a .f3d file on disk
  bool create(const std::string &filename, CreateMode cm = OverwriteMode);

  //! This routine is call if you want to write out global metadata to disk
  bool writeGlobalMetadata();

  //! This routine is called just before closing to write out any group
  //! membership to disk.
  bool writeGroupMembership();

 private:
  
  // Convenience methods -------------------------------------------------------

  //! Writes the mapping to the given hdf5 node.
  //! Mappings are assumed to be light-weight enough to be stored as 
  //! plain attributes under a group.
  bool writeMapping(hid_t partitionLocation, FieldMapping::Ptr mapping);
  
  //! Performs the actual writing of the layer to disk
  template <class Data_T>
  bool writeLayer(const std::string &partitionName, 
                  const std::string &layerName, 
                  bool isVectorLayer, 
                  typename Field<Data_T>::Ptr layer);

  //! Writes metadata for this layer
  bool writeMetadata(hid_t metadataGroup, FieldBase::Ptr layer);

  //! Writes metadata for this file
  bool writeMetadata(hid_t metadataGroup);

};

//----------------------------------------------------------------------------//
// Field3DInputFile-related callback functions
//----------------------------------------------------------------------------//

//! Namespace for file input specifics
namespace InputFile {

//! struct used to pass the class and partition info back to the 
//! parseLayers() callback
//! \ingroup file_int
struct ParseLayersInfo
{
  Field3DInputFile *file;
  std::string partitionName;
};

//! Gets called from readPartitionAndLayerInfo to check each group
//! found under the root of the file. It checks to see if it can
//! find a "partition" and then passes that to writePartition
herr_t parsePartitions(hid_t loc_id, const char *partitionName, 
                       const H5L_info_t *linfo, void *opdata);

//! Gets called from readPartitionAndLayerInfo to check each group
//! found under the root of the file. It checks to see if it can
//! find a "partition" and then passes that to writePartition
herr_t parseLayers(hid_t loc_id, const char *partitionName, 
                   const H5L_info_t *linfo, void *opdata);

} // namespace InputFile

//----------------------------------------------------------------------------//
// Field3DInputFile
//----------------------------------------------------------------------------//

template <class Data_T>
typename Field<Data_T>::Vec
Field3DInputFile::readScalarLayers(const std::string &name) const
{
  using namespace std;
  
  typedef typename Field<Data_T>::Ptr FieldPtr;
  typedef typename Field<Data_T>::Vec FieldList;

  FieldList ret;
  std::vector<std::string> parts;
  getIntPartitionNames(parts);

  for (vector<string>::iterator p = parts.begin(); p != parts.end(); ++p) {
    std::vector<std::string> layers;
    getIntScalarLayerNames(layers, *p);
    for (vector<string>::iterator l = layers.begin(); l != layers.end(); ++l) {
      // Only read if it matches the name
      if ((name.length() == 0) || (*l == name)) {
        FieldPtr mf = readScalarLayer<Data_T>(*p, *l);
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
Field3DInputFile::readScalarLayers(const std::string &partitionName, 
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
          FieldPtr mf = readScalarLayer<Data_T>(*p, *l);
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
typename Field<FIELD3D_VEC3_T<Data_T> >::Vec
Field3DInputFile::readVectorLayers(const std::string &name) const
{
  using namespace std;
  
  typedef typename Field<FIELD3D_VEC3_T<Data_T> >::Ptr FieldPtr;
  typedef typename Field<FIELD3D_VEC3_T<Data_T> >::Vec FieldList;
  
  FieldList ret;
  
  std::vector<std::string> parts;
  getIntPartitionNames(parts);
  
  for (vector<string>::iterator p = parts.begin(); p != parts.end(); ++p) {
    std::vector<std::string> layers;
    getIntVectorLayerNames(layers, *p);
    for (vector<string>::iterator l = layers.begin(); l != layers.end(); ++l) {
      // Only read if it matches the name
      if ((name.length() == 0) || (*l == name)) {
        FieldPtr mf = readVectorLayer<Data_T>(*p, *l);
        if (mf)
          ret.push_back(mf);
      }
    }
  }
  
  return ret;
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename Field<FIELD3D_VEC3_T<Data_T> >::Vec
Field3DInputFile::readVectorLayers(const std::string &partitionName, 
                                   const std::string &layerName) const
{
  using namespace std;
  
  typedef typename Field<FIELD3D_VEC3_T<Data_T> >::Ptr FieldPtr;
  typedef typename Field<FIELD3D_VEC3_T<Data_T> >::Vec FieldList;
  
  FieldList ret;

  if ((layerName.length() == 0) || (partitionName.length() == 0))
    return ret;
  
  std::vector<std::string> parts;
  getIntPartitionNames(parts);
  
  for (vector<string>::iterator p = parts.begin(); p != parts.end(); ++p) {
    std::vector<std::string> layers;
    getIntVectorLayerNames(layers, *p);
    if (removeUniqueId(*p) == partitionName) {
      for (vector<string>::iterator l = layers.begin(); 
           l != layers.end(); ++l) {
        // Only read if it matches the name
        if (*l == layerName) {
          FieldPtr mf = readVectorLayer<Data_T>(*p, *l);
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
typename Field<Data_T>::Ptr
Field3DInputFile::readLayer(const std::string &intPartitionName,
                            const std::string &layerName,
                            bool isVectorLayer) const
{
  using namespace boost;
  using namespace std;
  using namespace Hdf5Util;

  // Instantiate a null pointer for easier code reading
  typename Field<Data_T>::Ptr nullPtr;

  // Find the partition
  File::Partition::Ptr part = partition(intPartitionName);
  if (!part) {
    Msg::print(Msg::SevWarning, "Couldn't find partition: " + intPartitionName);
    return nullPtr;
  }

  // Find the layer in the partition
  const File::Layer *l;
  if (isVectorLayer)
    l = part->vectorLayer(layerName);
  else
    l = part->scalarLayer(layerName);
  if (!l) {
    Msg::print(Msg::SevWarning, "Couldn't find layer: " + layerName );
    return nullPtr;
  }

  // Open the layer group
  string layerPath = l->parent + "/" + l->name;
  H5ScopedGopen layerGroup(m_file, layerPath.c_str());

  if (layerGroup.id() < 0) {
    Msg::print(Msg::SevWarning, "Couldn't find layer group " + layerName 
              + " in .f3d file ");
    return nullPtr;
  }

  // Get the class name
  string className;
  if (!readAttribute(layerGroup.id(), "class_name", className)) {
    Msg::print(Msg::SevWarning, "Couldn't find class_name attrib in layer " + 
              layerName);
    return nullPtr;
  }
  

  // Construct the field and load the data
 
  typename Field<Data_T>::Ptr field;
  field = readField<Data_T>(className, layerGroup.id(), m_filename, layerPath);

  if (!field) {
#if 0 // This isn't really an error
    Msg::print(Msg::SevWarning, "Couldn't read the layer data of layer: " 
              + layerName);
#endif
    return nullPtr;
  }

  // read the metadata 
  string metadataPath = layerPath + "/metadata";
  H5ScopedGopen metadataGroup(m_file, metadataPath.c_str());
  if (metadataGroup.id() > 0) {    
    readMetadata(metadataGroup.id(), field);
  }

  // Set the name of the field so it's possible to re-create the file
  field->name = removeUniqueId(intPartitionName);
  field->attribute = layerName;
  field->setMapping(part->mapping);
  
  return field;
}

//----------------------------------------------------------------------------//

template <template <typename T> class Field_T, class Data_T>
typename Field_T<Data_T>::Vec
Field3DInputFile::readScalarLayersAs(const std::string &layerName) const
{
  typedef typename Field<Data_T>::Vec FieldList;
  typedef typename Field_T<Data_T>::Vec TypedFieldList;

  // First, read the layers as-is
  FieldList originals;
  originals = readScalarLayers<Data_T>(layerName);
  
  // Loop over fields, converting if needed
  TypedFieldList output;
  typename FieldList::iterator i = originals.begin();
  for (; i != originals.end(); ++i) {
    typename Field_T<Data_T>::Ptr targetField;
    targetField = field_dynamic_cast<Field_T<Data_T> >(*i);
    if (targetField) {
      output.push_back(targetField);
    } else {
      typename Field_T<Data_T>::Ptr newTarget(new Field_T<Data_T>);
      newTarget->name = (*i)->name;
      newTarget->attribute = (*i)->attribute;
      newTarget->copyMetadata(*i);
      newTarget->copyFrom(*i);
      output.push_back(newTarget);
    }
  }

  return output;
}

//----------------------------------------------------------------------------//

template <template <typename T> class Field_T, class Data_T>
typename Field_T<Data_T>::Vec
Field3DInputFile::readScalarLayersAs(const std::string &partitionName, 
                                     const std::string &layerName) const
{
  typedef typename Field<Data_T>::Vec FieldList;
  typedef typename Field_T<Data_T>::Vec TypedFieldList;

  // First, read the layers as-is
  FieldList originals;
  originals = readScalarLayers<Data_T>(partitionName, layerName);
  
  // Loop over fields, converting if needed
  TypedFieldList output;
  typename FieldList::iterator i = originals.begin();
  for (; i != originals.end(); ++i) {
    typename Field_T<Data_T>::Ptr targetField;
    targetField = field_dynamic_cast<Field_T<Data_T> >(*i);
    if (targetField) {
      output.push_back(targetField);
    } else {
      typename Field_T<Data_T>::Ptr newTarget(new Field_T<Data_T>);
      newTarget->name = (*i)->name;
      newTarget->attribute = (*i)->attribute;
      newTarget->copyMetadata(**i);
      newTarget->copyFrom(*i);
      output.push_back(newTarget);
    }
  }

  return output;
}

//----------------------------------------------------------------------------//

template <template <typename T> class Field_T, class Data_T>
typename Field_T<Data_T>::Vec
Field3DInputFile::readVectorLayersAs(const std::string &layerName) const
{
  typedef typename Field<Data_T>::Vec FieldList;
  typedef typename Field_T<Data_T>::Vec TypedFieldList;

  // First, read the layers as-is
  FieldList originals;
  originals = readVectorLayers<Data_T>(layerName);
  
  // Loop over fields, converting if needed
  TypedFieldList output;
  typename FieldList::iterator i = originals.begin();
  for (; i != originals.end(); ++i) {
    typename Field_T<Data_T>::Ptr targetField;
    targetField = field_dynamic_cast<Field_T<Data_T> >(*i);
    if (targetField) {
      output.push_back(targetField);
    } else {
      typename Field_T<Data_T>::Ptr newTarget(new Field_T<Data_T>);
      newTarget->name = (*i)->name;
      newTarget->attribute = (*i)->attribute;
      newTarget->copyMetadata(*i);
      newTarget->copyFrom(*i);
      output.push_back(newTarget);
    }
  }

  return output;
}

//----------------------------------------------------------------------------//

template <template <typename T> class Field_T, class Data_T>
typename Field_T<Data_T>::Vec
Field3DInputFile::readVectorLayersAs(const std::string &partitionName, 
                                     const std::string &layerName) const
{
  typedef typename Field<Data_T>::Vec FieldList;
  typedef typename Field_T<Data_T>::Vec TypedFieldList;

  // First, read the layers as-is
  FieldList originals;
  originals = readVectorLayers<Data_T>(partitionName, layerName);
  
  // Loop over fields, converting if needed
  TypedFieldList output;
  typename FieldList::iterator i = originals.begin();
  for (; i != originals.end(); ++i) {
    typename Field_T<Data_T>::Ptr targetField;
    targetField = field_dynamic_cast<Field_T<Data_T> >(*i);
    if (targetField) {
      output.push_back(targetField);
    } else {
      typename Field_T<Data_T>::Ptr newTarget(new Field_T<Data_T>);
      newTarget->name = (*i)->name;
      newTarget->attribute = (*i)->attribute;
      newTarget->copyMetadata(*i);
      newTarget->copyFrom(*i);
      output.push_back(newTarget);
    }
  }

  return output;
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
          if (isVectorLayer)
            layer = part->vectorLayer(layerName);
          else
            layer = part->scalarLayer(layerName);
          if (!layer) {
            Msg::print(Msg::SevWarning, "Couldn't find layer: " + layerName);
            return emptyList;
          }
          // Open the layer group
          string layerPath = layer->parent + "/" + layer->name;
          H5ScopedGopen layerGroup(m_file, layerPath.c_str());
          if (layerGroup.id() < 0) {
            Msg::print(Msg::SevWarning, "Couldn't find layer group " 
                      + layerName + " in .f3d file ");
            return emptyList;
          }
          // Read the extents and data window
          Box3i extents, dataW;
          if (!readAttribute(layerGroup, "extents", 6, extents.min.x)) {
            return emptyList;
          }
          if (!readAttribute(layerGroup, "data_window", 6, dataW.min.x)) {
            return emptyList;
          } 
          // Construct the field and load the data
          typename EmptyField<Data_T>::Ptr field(new EmptyField<Data_T>);
          field->setSize(extents, dataW);

          // read the metadata 
          string metadataPath = layerPath + "/metadata";
          H5ScopedGopen metadataGroup(m_file, metadataPath.c_str());
          if (metadataGroup.id() > 0) {    
            readMetadata(metadataGroup.id(), field);
          }

          // ... Set the name of the field so it's possible to 
          // ... re-create the file
          field->name = partitionName;
          field->attribute = layerName;
          field->setMapping(part->mapping);
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

template <class Data_T>
typename Field<Data_T>::Ptr
Field3DInputFile::readScalarLayer(const std::string &intPartitionName,
                                  const std::string &layerName) const
{
  return readLayer<Data_T>(intPartitionName, layerName, false);
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename Field<FIELD3D_VEC3_T<Data_T> >::Ptr
Field3DInputFile::readVectorLayer(const std::string &intPartitionName,
                                  const std::string &layerName) const
{
  return readLayer<FIELD3D_VEC3_T<Data_T> >(intPartitionName, layerName, true);
}

//----------------------------------------------------------------------------//
// Field3DOutputFile
//----------------------------------------------------------------------------//

template <class Data_T>
bool 
Field3DOutputFile::writeLayer(const std::string &userPartitionName, 
                              const std::string &layerName, 
                              bool isVectorLayer, 
                              typename Field<Data_T>::Ptr field)
{
  using namespace std;
  using namespace Exc;
  using namespace Hdf5Util;

  if (!field) {
    Msg::print(Msg::SevWarning, 
               "Called writeLayer with null pointer. Ignoring...");
    return false;
  }

  if (m_file < 0) {
    Msg::print(Msg::SevWarning, 
               "Attempting to write layer without opening file first. ");
    return false;
  }

  string partitionName = intPartitionName(userPartitionName, layerName, field);

  // See if the partition already exists or if we need to make it ---

  File::Partition::Ptr part = partition(partitionName);

  if (!part) {

    File::Partition::Ptr newPart(new File::Partition);

    newPart->name = partitionName;

    H5ScopedGcreate partGroup(m_file, newPart->name.c_str());
    if (partGroup.id() < 0) {
      Msg::print(Msg::SevWarning, 
                 "Error creating partition: " + newPart->name);
      return false;
    } 
    
    m_partitions.push_back(newPart);

    // Pick up new pointer
    part = partition(partitionName);

    // Add mapping group to the partition
    //! \todo We should probably remove the group on disk if we can't write
    //! the mapping
    try {
      if (!writeMapping(partGroup.id(), field->mapping())) {
        Msg::print(Msg::SevWarning, 
                  "writeMapping returned false for an unknown reason ");
        return false;
      }
    }
    catch (WriteMappingException &e) {
      Msg::print(Msg::SevWarning, "Couldn't write mapping for partition: " 
                + partitionName);
      return false;
    }
    catch (...) {
      Msg::print(Msg::SevWarning, 
                 "Unknown error when writing mapping for partition: " 
                 + partitionName);
      return false;
    }

    // Set the mapping of the partition. Since all layers share their 
    // partition's mapping, we can just pick this first one. All subsequent
    // additions to the same partition are checked to have the same mapping
    part->mapping = field->mapping();

    // Tag node as partition
    // Create a version attribute on the root node
    if (!writeAttribute(partGroup.id(), "is_field3d_partition", "1")) {
      Msg::print(Msg::SevWarning, "Adding partition string.");
      return false;
    }    

  } else {

    // If the partition already existed, we need to make sure that the layer
    // doesn't also exist
    if (!isVectorLayer) {
      if (part->scalarLayer(layerName)) {
        Msg::print(Msg::SevWarning, 
                  "Trying to add layer that already exists in file. Ignoring");
        return false;
      }
    } else {
      if (part->vectorLayer(layerName)) {
        Msg::print(Msg::SevWarning, 
                  "Trying to add layer that already exists in file. Ignoring");
        return false;
      }
    }
  }

  if (!field->mapping()) {
    Msg::print(Msg::SevWarning, 
              "Couldn't add layer \"" + layerName + "\" to partition \""
              + partitionName + "\" because the layer's mapping is null.");
    return false;    
  }

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
  H5ScopedGopen partGroup(m_file, part->name.c_str(), H5P_DEFAULT);

  // Build a Layer object ---

  File::Layer layer;
  layer.name = layerName;
  layer.parent = partitionName;

  // Add Layer to file ---

  H5ScopedGcreate layerGroup(partGroup.id(), layerName.c_str(),
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  if (layerGroup.id() < 0) {
    Msg::print(Msg::SevWarning, "Error creating layer: " + layerName);
    return false;
  }

  // Tag as layer
  if (!writeAttribute(layerGroup.id(), "class_type", "field3d_layer")) {
    Msg::print(Msg::SevWarning, "Error adding layer string.");
    return false;
  }    

  // Add metadata group and write it out  
  H5ScopedGcreate metadataGroup(layerGroup.id(), "metadata");
  if (metadataGroup.id() < 0) {
    Msg::print(Msg::SevWarning, "Error creating group: metadata");
    return false;
  }  
  if (!writeMetadata(metadataGroup.id(), field)) {
    Msg::print(Msg::SevWarning, "Error writing metadata.");
    return false;
  }    

  if (!writeField(layerGroup.id(), field)) {
    Msg::print(Msg::SevWarning, "Error writing layer: " + layer.name);
    return false;
  }
  
  // Add layer to partition ---

  if (isVectorLayer)
    part->addVectorLayer(layer);
  else
    part->addScalarLayer(layer);

  return true;  
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool 
Field3DOutputFile::writeScalarLayer(const std::string &partitionName, 
                                    const std::string &layerName, 
                                    typename Field<Data_T>::Ptr field)
{
  return writeLayer<Data_T>(partitionName, layerName, false, field);
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool 
Field3DOutputFile::writeScalarLayer(typename Field<Data_T>::Ptr layer)
{
  if (layer->name.size() == 0) {
    Msg::print(Msg::SevWarning, "Field3DOutputFile::writeScalarLayer: "
               "Tried to write a scalar layer with no name");
    return false;
  }
  if (layer->attribute.size() == 0) {
    Msg::print(Msg::SevWarning, "Field3DOutputFile::writeScalarLayer: "
               "Tried to write a scalar layer with no attribute name");
    return false;
  }
  return writeScalarLayer<Data_T>(layer->name, layer->attribute, layer);
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool 
Field3DOutputFile::
writeVectorLayer(const std::string &partitionName, 
                 const std::string &layerName, 
                 typename Field<FIELD3D_VEC3_T<Data_T> >::Ptr field)
{
  return writeLayer<FIELD3D_VEC3_T<Data_T> >(partitionName, layerName, 
                                             true, field);
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool 
Field3DOutputFile::writeVectorLayer
  (typename Field<FIELD3D_VEC3_T<Data_T> >::Ptr layer)
{
  if (layer->name.size() == 0) {
    Msg::print(Msg::SevWarning, "Field3DOutputFile::writeVectorLayer: "
               "Tried to write a vector layer with no name");
    return false;
  }
  if (layer->attribute.size() == 0) {
    Msg::print(Msg::SevWarning, "Field3DOutputFile::writeVectorLayer: "
               "Tried to write a vector layer with no attribute name");
    return false;
  }
  return writeVectorLayer<Data_T>(layer->name, layer->attribute, layer);
}

//----------------------------------------------------------------------------//
// Template Function Implementations
//----------------------------------------------------------------------------//

template <class Data_T>
typename Field<Data_T>::Ptr 
readField(const std::string &className, hid_t layerGroup,
          const std::string &filename, const std::string &layerPath)
{

  ClassFactory &factory = ClassFactory::singleton();
  
  typedef typename Field<Data_T>::Ptr FieldPtr;

  FieldIO::Ptr io = factory.createFieldIO(className);
  assert(io != 0);
  if (!io) {
    Msg::print(Msg::SevWarning, "Unable to find class type: " + 
               className);
    return FieldPtr();
  }

  DataTypeEnum typeEnum = DataTypeTraits<Data_T>::typeEnum();
  FieldBase::Ptr field = io->read(layerGroup, filename, layerPath, typeEnum);
  
  if (!field) {
    // We don't need to print a message, because it could just be that
    // a layer of the specified data type and name couldn't be found
    return FieldPtr();
  }

  FieldPtr result = field_dynamic_cast<Field<Data_T> >(field);

  if (result)
    return result;

  return FieldPtr();
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif
