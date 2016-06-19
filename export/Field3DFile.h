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

#include <boost/shared_ptr.hpp>

#include "EmptyField.h"
#include "Field.h"
#include "Field3DFileHDF5.h"
#include "FieldMetadata.h"
#include "ClassFactory.h"
#include "OgawaFwd.h"

//----------------------------------------------------------------------------//

#include "ns.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Forward declarations
//----------------------------------------------------------------------------//

class Field3DInputFileHDF5;
class Field3DOutputFileHDF5;

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

class FIELD3D_API Partition : public RefBase
{
public:

  typedef std::vector<Layer> LayerList;

  typedef boost::intrusive_ptr<Partition> Ptr;
  typedef boost::intrusive_ptr<const Partition> CPtr;

  // RTTI replacement ----------------------------------------------------------

  typedef Partition class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;   
  
  static const char *staticClassType()
  {
    return "Partition";
  }

  // Ctors, dtor ---------------------------------------------------------------

  //! Ctor
  Partition() 
    : RefBase() 
  { }

  // From RefBase --------------------------------------------------------------

  //! \name From RefBase
  //! \{

  virtual std::string className() const;
  
  //! \}
  
  // Main methods --------------------------------------------------------------

  //! Adds a layer
  void addLayer(const File::Layer &layer);

  //! Finds a layer
  const File::Layer* layer(const std::string &name) const;
  
  //! Gets all the layer names. 
  void getLayerNames(std::vector<std::string> &names) const;

  //! Returns a reference to the OgOGroup
  OgOGroup& group() const;
  //! Sets the group pointer
  void setGroup(boost::shared_ptr<OgOGroup> ptr);

  // Public data members -------------------------------------------------------

  //! Name of the partition
  std::string name;
  //! Pointer to the mapping object.
  FieldMapping::Ptr mapping;

private:

  // Private data members ------------------------------------------------------

  //! The layers belonging to this partition
  LayerList m_layers;
  //! Group representing the partition
  boost::shared_ptr<OgOGroup> m_group;

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef RefBase base;

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

class FIELD3D_API Field3DFileBase : public MetadataCallback
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

  //! \}

  //! \name Convenience methods for partitionName
  //! \{

  //! Add to the group membership
  void addGroupMembership(const GroupMembershipMap &groupMembers);

  //! \}

  // Access to metadata --------------------------------------------------------

  //! accessor to the m_metadata class
  FieldMetadata& metadata()
  { 
    if (m_hdf5Base) {
      return m_hdf5Base->metadata();
    }
    return m_metadata; 
  }

  //! Read only access to the m_metadata class
  const FieldMetadata& metadata() const
  { 
    if (m_hdf5Base) {
      return m_hdf5Base->metadata();
    }
    return m_metadata; 
  }
 
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

  //! Returns a pointer to the given partition
  //! \returns NULL if no partition was found of that name
  File::Partition::Ptr getPartition(const std::string &partitionName) const
  { return partition(partitionName); }

  //! Closes the file if open.
  virtual void closeInternal() = 0;
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

  //! Returns a unique partition name given the requested name. This ensures
  //! that partitions with matching mappings get the same name but each
  //! subsequent differing mapping gets a new, separate name
  std::string intPartitionName(const std::string &partitionName,
                               const std::string &layerName,
                               FieldRes::Ptr field);

  //! Strips any unique identifiers from the partition name and returns
  //! the original name
  std::string removeUniqueId(const std::string &partitionName) const;

  //! \}

  // Data members --------------------------------------------------------------

  //! This stores layer info
  std::vector<LayerInfo> m_layerInfo;

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
  FieldMetadata m_metadata;

  //! HDF5 fallback
  boost::shared_ptr<Field3DFileHDF5Base> m_hdf5Base;

private:

  // Private member functions --------------------------------------------------

  Field3DFileBase(const Field3DFileBase&);
  void operator =(const Field3DFileBase&); 

};

//----------------------------------------------------------------------------//
// Field3DInputFile
//----------------------------------------------------------------------------//

/*! \class Field3DInputFile
  \brief Provides reading of .f3d (internally, hdf5 or Ogawa) files.
  \ingroup file

  Refer to \ref using_files for examples of how to use this in your code.

 */

//----------------------------------------------------------------------------//

class FIELD3D_API Field3DInputFile : public Field3DFileBase 
{
public:

  // Ctors, dtor ---------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  Field3DInputFile();
  virtual ~Field3DInputFile();

  //! \}

  // Main interface ------------------------------------------------------------

  //! Opens the given file
  //! \returns Whether successful
  bool open(const std::string &filename);

  //! Returns an encoding descriptor of the given file 
  const std::string &encoding() const
  { 
    const static std::string encodings[2] = { "Ogawa", "HDF5" };
    return encodings[m_hdf5 ? 1 : 0];
  }

  //! \name Reading layers from disk
  //! \{

  template <class Data_T>
  typename Field<Data_T>::Vec
  readLayers(const std::string &layerName = std::string("")) const;

  template <class Data_T>
  typename Field<Data_T>::Vec
  readLayers(const std::string &partitionName,
             const std::string &layerName) const;

  //! \name Backward compatibility
  //! \{

  //! Retrieves all the layers of scalar type and maintains their on-disk
  //! data types
  //! \param layerName If a string is passed in, only layers of that name will
  //! be read from disk.
  template <class Data_T>
  typename Field<Data_T>::Vec
  readScalarLayers(const std::string &layerName = std::string("")) const
  { 
    if (m_hdf5) {
      return m_hdf5->readScalarLayers<Data_T>(layerName);
    }
    return readLayers<Data_T>(layerName); 
  }

  //! This one allows the allows the partitionName to be passed in
  template <class Data_T>
  typename Field<Data_T>::Vec
  readScalarLayers(const std::string &partitionName, 
                   const std::string &layerName) const
  { 
    if (m_hdf5) {
      return m_hdf5->readScalarLayers<Data_T>(partitionName, layerName);
    }
    return readLayers<Data_T>(partitionName, layerName); 
  }

  //! Retrieves all the layers of vector type and maintains their on-disk
  //! data types
  //! \param layerName If a string is passed in, only layers of that name will
  //! be read from disk.
  template <class Data_T>
  typename Field<FIELD3D_VEC3_T<Data_T> >::Vec
  readVectorLayers(const std::string &layerName = std::string("")) const
  { 
    if (m_hdf5) {
      return m_hdf5->readVectorLayers<Data_T>(layerName);
    }
    return readLayers<FIELD3D_VEC3_T<Data_T> >(layerName); 
  }

  //! This version allows you to pass in the partition name
  template <class Data_T>
  typename Field<FIELD3D_VEC3_T<Data_T> >::Vec
  readVectorLayers(const std::string &partitionName, 
                   const std::string &layerName) const
  { 
    if (m_hdf5) {
      return m_hdf5->readVectorLayers<Data_T>(partitionName, layerName);
    }
    return readLayers<FIELD3D_VEC3_T<Data_T> >(partitionName, layerName); 
  }

  //! \}

  //! \name Reading proxy data from disk
  //! \{

  //! Retrieves a proxy version (EmptyField) of each layer .
  //! \note Although the call is templated, all fields are read, regardless
  //! of bit depth.
  //! \param name If a string is passed in, only layers of that name will
  //! be read from disk.
  template <class Data_T>
  typename EmptyField<Data_T>::Vec
  readProxyLayer(const std::string &partitionName, 
                 const std::string &layerName, 
                 bool isVectorLayer) const;

  //! Retrieves a proxy version (EmptyField) of each scalar layer 
  //! \note Although the call is templated, all fields are read, regardless
  //! of bit depth.
  //! \param name If a string is passed in, only layers of that name will
  //! be read from disk.
  template <class Data_T>
  typename EmptyField<Data_T>::Vec
  readProxyScalarLayers(const std::string &name = std::string("")) const;

  //! Retrieves a proxy version (EmptyField) of each vector layer 
  //! \note Although the call is templated, all fields are read, regardless
  //! of bit depth.
  //! \param name If a string is passed in, only layers of that name will
  //! be read from disk.
  template <class Data_T>
  typename EmptyField<Data_T>::Vec
  readProxyVectorLayers(const std::string &name = std::string("")) const;

  //! \}

private:

  // From Field3DFileBase ------------------------------------------------------

  virtual void closeInternal()
  {
    if (m_hdf5) {
      m_hdf5->closeInternal();
      return;
    }

    cleanup();
  }

  void cleanup()
  {
    // The destruction of the various Ogawa components must happen in the
    // right order
    
    // First, the partition groups
    m_partitions.clear();
    // Then the root group
    m_root.reset();
    // Finally, the archive
    m_archive.reset();
  }

  // Convenience methods -------------------------------------------------------

  //! This call does the actual reading of a layer. Notice that it expects
  //! a unique -internal- partition name.
  template <class Data_T>
  typename Field<Data_T>::Ptr 
  readLayer(const std::string &intPartitionName, 
            const std::string &layerName) const;

  //! Retrieves a proxy version (EmptyField) from a given Ogawa location
  //! \note Although the call is templated, all fields are read, regardless
  //! of bit depth.
  template <class Data_T>
  typename EmptyField<Data_T>::Ptr
  readProxyLayer(OgIGroup &location, const std::string &name,
                 const std::string &attribute, 
                 FieldMapping::Ptr mapping) const;
  
  //! Sets up all the partitions and layers, but does not load any data
  bool readPartitionAndLayerInfo();

  //! Read metadata for this layer
  bool readMetadata(const OgIGroup &metadataGroup, FieldBase::Ptr field) const;

  //! Read global metadata for this file
  bool readMetadata(const OgIGroup &metadataGroup);

  // Data members --------------------------------------------------------------

  //! Filename, only to be set by open().
  std::string m_filename;
  //! Pointer to the Ogawa archive
  boost::shared_ptr<Alembic::Ogawa::IArchive> m_archive;
  //! Pointer to root group
  boost::shared_ptr<OgIGroup> m_root;

  //! HDF5 fallback
  boost::shared_ptr<Field3DInputFileHDF5> m_hdf5;

};

//----------------------------------------------------------------------------//
// Utility functions
//----------------------------------------------------------------------------//

/*! \brief checks to see if a file/directory exists or not
  \param[in] filename the file/directory to check
  \retval true if it exists
  \retval false if it does not exist
*/
bool fileExists(const std::string &filename);

//----------------------------------------------------------------------------//
// Field3DOutputFile
//----------------------------------------------------------------------------//

/*! \class Field3DOutputFile
  \ingroup file
  \brief Provides writing of .f3d (internally, hdf5 or Ogawa) files.

  Refer to \ref using_files for examples of how to use this in your code.

 */

//----------------------------------------------------------------------------//

class FIELD3D_API Field3DOutputFile : public Field3DFileBase 
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

  //! Creates a .f3d file on disk
  bool create(const std::string &filename, CreateMode cm = OverwriteMode);

  //! Whether to output ogawa files
  static void useOgawa(const bool enabled)
  { 
    // simple temporary endian check
    union {
      uint32_t l;
      char c[4];
    } u;
    
    u.l = 0x01234567;
    
    if (u.c[0] == 0x67) {
      ms_doOgawa = enabled;
    } else {
      std::cerr << "WARNING: Field3D only supports Ogawa-backed files "
                << "on little-endian systems." << std::endl;
      ms_doOgawa = false;
    }
  }

  //! \name Writing layer to disk
  //! \{

  //! Writes a scalar layer to the "Default" partition.
  template <class Data_T>
  bool writeLayer(const std::string &layerName, 
                  typename Field<Data_T>::Ptr layer)
  { 
    return writeLayer<Data_T>(std::string("default"), layerName, layer); 
  }

  //! Writes a layer to a specific partition. The partition will be created if
  //! not specified.
  template <class Data_T>
  bool writeLayer(const std::string &partitionName, 
                  const std::string &layerName, 
                  typename Field<Data_T>::Ptr layer);

  //! Writes a layer to a specific partition. The field name and attribute
  //! name are used for partition and layer, respectively
  template <class Data_T>
  bool writeLayer(typename Field<Data_T>::Ptr layer)
  { 
    return writeLayer<Data_T>(layer->name, layer->attribute, layer); 
  }

  //! \}

  //! \name Backward compatibility
  //! \{

  //! Writes a scalar layer to the "Default" partition.
  template <class Data_T>
  bool writeScalarLayer(const std::string &layerName, 
                        typename Field<Data_T>::Ptr layer)
  { 
    if (m_hdf5) {
      return m_hdf5->writeScalarLayer<Data_T>(layerName, layer);
    }
    return writeScalarLayer<Data_T>(std::string("default"), layerName, layer); 
  }

  //! Writes a layer to a specific partition. The partition will be created if
  //! not specified.
  template <class Data_T>
  bool writeScalarLayer(const std::string &partitionName, 
                        const std::string &layerName, 
                        typename Field<Data_T>::Ptr layer)
  { 
    if (m_hdf5) {
      return m_hdf5->writeScalarLayer<Data_T>(partitionName, layerName, layer);
    }
    return writeLayer<Data_T>(partitionName, layerName, layer); 
  }

  //! Writes a layer to a specific partition. The field name and attribute
  //! name are used for partition and layer, respectively
  template <class Data_T>
  bool writeScalarLayer(typename Field<Data_T>::Ptr layer)
  { 
    if (m_hdf5) {
      return m_hdf5->writeScalarLayer<Data_T>(layer);
    }
    return writeLayer<Data_T>(layer); 
  }

  //! Writes a scalar layer to the "Default" partition.
  template <class Data_T>
  bool writeVectorLayer(const std::string &layerName, 
                        typename Field<FIELD3D_VEC3_T<Data_T> >::Ptr layer)
  { 
    if (m_hdf5) {
      return m_hdf5->writeVectorLayer<Data_T>(layerName, layer);
    }
    return writeVectorLayer<Data_T>(std::string("default"), layerName, layer); 
  }

  //! Writes a layer to a specific partition. The partition will be created if
  //! not specified.
  template <class Data_T>
  bool writeVectorLayer(const std::string &partitionName, 
                        const std::string &layerName, 
                        typename Field<FIELD3D_VEC3_T<Data_T> >::Ptr layer)
  { 
    if (m_hdf5) {
      return m_hdf5->writeVectorLayer<Data_T>(partitionName, layerName, layer);
    }
    return writeLayer<FIELD3D_VEC3_T<Data_T> >(partitionName, layerName, layer);
  }

  //! Writes a layer to a specific partition. The field name and attribute
  //! name are used for partition and layer, respectively
  template <class Data_T>
  bool writeVectorLayer(typename Field<FIELD3D_VEC3_T<Data_T> >::Ptr layer)
  { 
    if (m_hdf5) {
      return m_hdf5->writeVectorLayer<Data_T>(layer);
    }
    return writeLayer<FIELD3D_VEC3_T<Data_T> >(layer); 
  }

  //! This routine is call if you want to write out global metadata to disk
  bool writeGlobalMetadata();

  //! This routine is called just before closing to write out any group
  //! membership to disk.
  bool writeGroupMembership();

   //! \}

private:
  
  // From Field3DFileBase ------------------------------------------------------

  virtual void closeInternal()
  {
    if (m_hdf5) {
      m_hdf5->closeInternal();
      return;
    }
    cleanup();
  }

  void cleanup()
  {
    // The destruction of the various Ogawa components must happen in the
    // right order
    
    // First, the partition groups
    m_partitions.clear();
    // Then the root group
    m_root.reset();
    // Finally, the archive
    m_archive.reset();
  }

  // Convenience methods -------------------------------------------------------

  //! Increment the partition or make it zero if there's not an integer suffix
  std::string incrementPartitionName(std::string &pname);

  //! Create newPartition given the input config
  File::Partition::Ptr
  createNewPartition(const std::string &partitionName,
                     const std::string &layerName, FieldRes::Ptr field);
  //! Writes the mapping to the given Og node.
  //! Mappings are assumed to be light-weight enough to be stored as 
  //! plain attributes under a group.
  bool writeMapping(OgOGroup &partitionGroup, FieldMapping::Ptr mapping);
  
  //! Writes metadata for this layer
  bool writeMetadata(OgOGroup &metadataGroup, FieldBase::Ptr layer);

  //! Writes metadata for this file
  bool writeMetadata(OgOGroup &metadataGroup);

 // Data members --------------------------------------------------------------

  //! Whether to output ogawa files
  static bool ms_doOgawa;

  //! Pointer to the Ogawa archive
  boost::shared_ptr<Alembic::Ogawa::OArchive> m_archive;
  //! Pointer to root group
  boost::shared_ptr<OgOGroup> m_root;

  //! HDF5 fallback
  boost::shared_ptr<Field3DOutputFileHDF5> m_hdf5;

};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif
