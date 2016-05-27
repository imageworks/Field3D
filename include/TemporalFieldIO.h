//----------------------------------------------------------------------------//

#ifndef _INCLUDED_TemporalFieldIO_H_
#define _INCLUDED_TemporalFieldIO_H_

//----------------------------------------------------------------------------//

#include <vector>

#include "FieldIO.h"
#include "Field3DFile.h"
#include "OgIO.h"
#include "TemporalField.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// TemporalFieldIO
//----------------------------------------------------------------------------//

/*! \class TemporalFieldIO
   Defines the IO for a TemporalField object
 */

//----------------------------------------------------------------------------//

class TemporalFieldIO : public FieldIO 
{

public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<TemporalFieldIO> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef TemporalFieldIO class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;

  const char *staticClassType() const
  {
    return "TemporalFieldIO";
  }
  
  // Constructors --------------------------------------------------------------

  //! Ctor
  TemporalFieldIO() 
   : FieldIO()
  { }

  //! Dtor
  virtual ~TemporalFieldIO() 
  { /* Empty */ }


  static FieldIO::Ptr create()
  { return Ptr(new TemporalFieldIO); }

  // From FieldIO --------------------------------------------------------------

  //! Reads the field at the given location and tries to create a TemporalField
  //! object from it.
  //! \returns Null if no object was read
  virtual FieldBase::Ptr read(hid_t layerGroup, const std::string &filename, 
                              const std::string &layerPath,
                              DataTypeEnum typeEnum);

  //! Reads the field at the given location and tries to create a TemporalField
  //! object from it.
  //! \returns Null if no object was read
  virtual FieldBase::Ptr read(const OgIGroup &layerGroup, 
                              const std::string &filename, 
                              const std::string &layerPath,
                              OgDataType typeEnum);

  //! Writes the given field to disk. 
  //! \return true if successful, otherwise false
  virtual bool write(hid_t layerGroup, FieldBase::Ptr field);

  //! Writes the given field to disk. 
  //! \return true if successful, otherwise false
  virtual bool write(OgOGroup &layerGroup, FieldBase::Ptr field);

  //! Returns the class name
  virtual std::string className() const
  { return "TemporalField"; }

private:

  // Internal methods ----------------------------------------------------------

  //! This call writes all the attributes and sets up the data space.
  template <class Data_T>
  bool writeInternal(hid_t layerGroup, 
                     typename TemporalField<Data_T>::Ptr field);

  //! This call writes all the attributes and sets up the data space.
  template <class Data_T>
  bool writeInternal(OgOGroup &layerGroup, 
                     typename TemporalField<Data_T>::Ptr field);

  //! Reads the data that is dependent on the data type on disk
  template <class Data_T>
  bool readData(hid_t location, 
                int numBlocks, 
                const std::string &filename, 
                const std::string &layerPath, 
                typename TemporalField<Data_T>::Ptr result);

  template <class Data_T>
  typename TemporalField<Data_T>::Ptr
  readData(const OgIGroup &location, const Box3i &extents, 
           const Box3i &dataWindow, const size_t blockOrder, 
           const size_t numBlocks, const std::string &filename, 
           const std::string &layerPath);

  // Strings -------------------------------------------------------------------

  static const int         k_versionNumber;
  static const std::string k_versionAttrName;
  static const std::string k_extentsStr;
  static const std::string k_extentsMinStr;
  static const std::string k_extentsMaxStr;
  static const std::string k_dataWindowStr;
  static const std::string k_dataWindowMinStr;
  static const std::string k_dataWindowMaxStr;
  static const std::string k_componentsStr;
  static const std::string k_blockOrderStr;
  static const std::string k_numBlocksStr;
  static const std::string k_blockResStr;
  static const std::string k_bitsPerComponentStr;
  static const std::string k_numOccupiedBlocksStr;
  static const std::string k_offsetDataStr;
  static const std::string k_timeDataStr;
  static const std::string k_valueDataStr;
  
  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef FieldIO base;  

};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // include guard
