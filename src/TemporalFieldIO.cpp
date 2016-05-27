//----------------------------------------------------------------------------//

#include "TemporalFieldIO.h"

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

#include "InitIO.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Using
//----------------------------------------------------------------------------//

using std::cout;
using std::endl;

//----------------------------------------------------------------------------//
// Field3D namespaces
//----------------------------------------------------------------------------//

using namespace Exc;
using namespace Hdf5Util;

//----------------------------------------------------------------------------//
// Static members
//----------------------------------------------------------------------------//

const int         TemporalFieldIO::k_versionNumber(1);
const std::string TemporalFieldIO::k_versionAttrName("version");
const std::string TemporalFieldIO::k_extentsStr("extents");
const std::string TemporalFieldIO::k_extentsMinStr("extents_min");
const std::string TemporalFieldIO::k_extentsMaxStr("extents_max");
const std::string TemporalFieldIO::k_dataWindowStr("data_window");
const std::string TemporalFieldIO::k_dataWindowMinStr("data_window_min");
const std::string TemporalFieldIO::k_dataWindowMaxStr("data_window_max");
const std::string TemporalFieldIO::k_componentsStr("components");
const std::string TemporalFieldIO::k_offsetDataStr("offset_data");
const std::string TemporalFieldIO::k_timeDataStr("time_data");
const std::string TemporalFieldIO::k_valueDataStr("value_data");
const std::string TemporalFieldIO::k_blockOrderStr("block_order");
const std::string TemporalFieldIO::k_numBlocksStr("num_blocks");
const std::string TemporalFieldIO::k_blockResStr("block_res");
const std::string TemporalFieldIO::k_bitsPerComponentStr("bits_per_component");
const std::string TemporalFieldIO::k_numOccupiedBlocksStr("num_occupied_blocks");

//----------------------------------------------------------------------------//

FieldBase::Ptr
TemporalFieldIO::read(hid_t layerGroup, const std::string &filename, 
                      const std::string &layerPath,
                      DataTypeEnum typeEnum)
{
  Box3i extents, dataW;
  int components;
  int blockOrder;
  int numBlocks;
  V3i blockRes;
  
  if (layerGroup == -1) {
    Msg::print(Msg::SevWarning, "Bad layerGroup.");
    return FieldBase::Ptr();
  }

  int version;
  if (!readAttribute(layerGroup, k_versionAttrName, 1, version)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_versionAttrName);

  if (version != k_versionNumber) 
    throw UnsupportedVersionException("TemporalField version not supported: " +
                                      boost::lexical_cast<std::string>(version));

  if (!readAttribute(layerGroup, k_extentsStr, 6, extents.min.x)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_extentsStr);

  if (!readAttribute(layerGroup, k_dataWindowStr, 6, dataW.min.x)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_dataWindowStr);
  
  if (!readAttribute(layerGroup, k_componentsStr, 1, components)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_componentsStr);
  
  // Read block order
  if (!readAttribute(layerGroup, k_blockOrderStr, 1, blockOrder)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_blockOrderStr);

  // Read number of blocks total
  if (!readAttribute(layerGroup, k_numBlocksStr, 1, numBlocks)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_numBlocksStr);

  // Read block resolution in each dimension
  if (!readAttribute(layerGroup, k_blockResStr, 3, blockRes.x)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_blockResStr);

  // ... Check that it matches the # reported by summing the active blocks

  int numCalculatedBlocks = blockRes.x * blockRes.y * blockRes.z;
  if (numCalculatedBlocks != numBlocks)
    throw FileIntegrityException("Incorrect block count in TemporalFieldIO::read");

  // Call the appropriate read function based on the data type ---

  FieldBase::Ptr result;
  
  int occupiedBlocks;
  if (!readAttribute(layerGroup, k_numOccupiedBlocksStr, 1, occupiedBlocks)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_numOccupiedBlocksStr);

  // Check the data type ---

  int bits;
  if (!readAttribute(layerGroup, k_bitsPerComponentStr, 1, bits)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_bitsPerComponentStr);  

  bool isHalf = false;
  bool isFloat = false;
  bool isDouble = false;

  switch (bits) {
  case 16:
    isHalf = true;
    break;
  case 64:
    isDouble = true;
    break;
  case 32:
  default:
    isFloat = true;
  }

  // Finally, read the data ---

  if (components == 1) {
    if (isHalf && typeEnum == DataTypeHalf) {
      TemporalField<half>::Ptr field(new TemporalField<half>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<half>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    } else if (isFloat && typeEnum == DataTypeFloat) {
      TemporalField<float>::Ptr field(new TemporalField<float>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<float>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    } else if (isDouble && typeEnum == DataTypeDouble) {
      TemporalField<double>::Ptr field(new TemporalField<double>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<double>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    }
  } else if (components == 3) {
    if (isHalf && typeEnum == DataTypeVecHalf) {
      TemporalField<V3h>::Ptr field(new TemporalField<V3h>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<V3h>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    } else if (isFloat && typeEnum == DataTypeVecFloat) {
      TemporalField<V3f>::Ptr field(new TemporalField<V3f>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<V3f>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    } else if (isDouble && typeEnum == DataTypeVecDouble) {
      TemporalField<V3d>::Ptr field(new TemporalField<V3d>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<V3d>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    }    
  }

  return result;
}

//----------------------------------------------------------------------------//

FieldBase::Ptr 
TemporalFieldIO::read(const OgIGroup &layerGroup, const std::string &filename, 
                      const std::string &layerPath, OgDataType typeEnum)
{
  Box3i extents, dataW;
  int blockOrder;
  int numBlocks;
  V3i blockRes;
  
  if (!layerGroup.isValid()) {
    throw MissingGroupException("Invalid group in TemporalFieldIO::read()");
  }

  // Check version ---

  OgIAttribute<int> versionAttr = 
    layerGroup.findAttribute<int>(k_versionAttrName);
  if (!versionAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_versionAttrName);
  }
  const int version = versionAttr.value();

  if (version != k_versionNumber) {
    throw UnsupportedVersionException("TemporalField version not supported: " +
                                      boost::lexical_cast<std::string>(version));
  }

  // Get extents ---

  OgIAttribute<veci32_t> extMinAttr = 
    layerGroup.findAttribute<veci32_t>(k_extentsMinStr);
  OgIAttribute<veci32_t> extMaxAttr = 
    layerGroup.findAttribute<veci32_t>(k_extentsMaxStr);
  if (!extMinAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_extentsMinStr);
  }
  if (!extMaxAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_extentsMaxStr);
  }

  extents.min = extMinAttr.value();
  extents.max = extMaxAttr.value();

  // Get data window ---

  OgIAttribute<veci32_t> dwMinAttr = 
    layerGroup.findAttribute<veci32_t>(k_dataWindowMinStr);
  OgIAttribute<veci32_t> dwMaxAttr = 
    layerGroup.findAttribute<veci32_t>(k_dataWindowMaxStr);
  if (!dwMinAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_dataWindowMinStr);
  }
  if (!dwMaxAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_dataWindowMaxStr);
  }

  dataW.min = dwMinAttr.value();
  dataW.max = dwMaxAttr.value();

  // Get num components ---

  OgIAttribute<uint8_t> numComponentsAttr = 
    layerGroup.findAttribute<uint8_t>(k_componentsStr);
  if (!numComponentsAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_componentsStr);
  }

  // Read block order ---

  OgIAttribute<uint8_t> blockOrderAttr = 
    layerGroup.findAttribute<uint8_t>(k_blockOrderStr); 
  if (!blockOrderAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_blockOrderStr);
  }
  blockOrder = blockOrderAttr.value();

  // Read number of blocks total ---
  
  OgIAttribute<uint32_t> numBlocksAttr = 
    layerGroup.findAttribute<uint32_t>(k_numBlocksStr);
  if (!numBlocksAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_numBlocksStr);
  }
  numBlocks = numBlocksAttr.value();

  // Read block resolution in each dimension ---

  OgIAttribute<veci32_t> blockResAttr = 
    layerGroup.findAttribute<veci32_t>(k_blockResStr);
  if (!blockResAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_blockResStr);
  }
  blockRes = blockResAttr.value();

  // ... Check that it matches the # reported by summing the active blocks

  int numCalculatedBlocks = blockRes.x * blockRes.y * blockRes.z;
  if (numCalculatedBlocks != numBlocks) {
    throw FileIntegrityException("Incorrect block count in "
                                 "TemporalFieldIO::read()");
  }

  // Call the appropriate read function based on the data type ---

  OgIAttribute<uint32_t> occupiedBlocksAttr = 
    layerGroup.findAttribute<uint32_t>(k_numOccupiedBlocksStr);
  if (!occupiedBlocksAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_numOccupiedBlocksStr);
  }

  // Finally, read the data ---

  FieldBase::Ptr result;
  
  OgDataType typeOnDisk = layerGroup.compressedDatasetType(k_valueDataStr);

  if (typeEnum == typeOnDisk) {
    if (typeEnum == F3DFloat16) {
      result = readData<float16_t>(layerGroup, extents, dataW, blockOrder,
                                   numBlocks, filename, layerPath);
    } else if (typeEnum == F3DFloat32) {
      result = readData<float32_t>(layerGroup, extents, dataW, blockOrder,
                                   numBlocks, filename, layerPath);
    } else if (typeEnum == F3DFloat64) {
      result = readData<float64_t>(layerGroup, extents, dataW, blockOrder,
                                   numBlocks, filename, layerPath);
    } else if (typeEnum == F3DVec16) {
      result = readData<vec16_t>(layerGroup, extents, dataW, blockOrder,
                                 numBlocks, filename, layerPath);
    } else if (typeEnum == F3DVec32) {
      result = readData<vec32_t>(layerGroup, extents, dataW, blockOrder,
                                 numBlocks, filename, layerPath);
    } else if (typeEnum == F3DVec64) {
      result = readData<vec64_t>(layerGroup, extents, dataW, blockOrder,
                                 numBlocks, filename, layerPath);
    } 
  }

  return result;
}

//----------------------------------------------------------------------------//

bool
TemporalFieldIO::write(hid_t layerGroup, FieldBase::Ptr field)
{
  if (layerGroup == -1) {
    Msg::print(Msg::SevWarning, "Bad layerGroup.");
    return false;
  }

  // Add version attribute
  if (!writeAttribute(layerGroup, k_versionAttrName, 
                      1, k_versionNumber)) {
    Msg::print(Msg::SevWarning, "Error adding version attribute.");
    return false;
  }

  TemporalField<half>::Ptr halfField = 
    field_dynamic_cast<TemporalField<half> >(field);
  TemporalField<float>::Ptr floatField = 
    field_dynamic_cast<TemporalField<float> >(field);
  TemporalField<double>::Ptr doubleField = 
    field_dynamic_cast<TemporalField<double> >(field);
  TemporalField<V3h>::Ptr vecHalfField = 
    field_dynamic_cast<TemporalField<V3h> >(field);
  TemporalField<V3f>::Ptr vecFloatField = 
    field_dynamic_cast<TemporalField<V3f> >(field);
  TemporalField<V3d>::Ptr vecDoubleField = 
    field_dynamic_cast<TemporalField<V3d> >(field);

  bool success = true;
  if (halfField) {
    success = writeInternal<half>(layerGroup, halfField);
  } else if (floatField) {
    success = writeInternal<float>(layerGroup, floatField);
  } else if (doubleField) {
    success = writeInternal<double>(layerGroup, doubleField);
  } else if (vecHalfField) {
    success = writeInternal<V3h>(layerGroup, vecHalfField);
  } else if (vecFloatField) {
    success = writeInternal<V3f>(layerGroup, vecFloatField);
  } else if (vecDoubleField) {
    success = writeInternal<V3d>(layerGroup, vecDoubleField);
  } else {
    throw WriteLayerException("TemporalFieldIO::write does not support the given "
                              "TemporalField template parameter");
  }

  return success;
}

//----------------------------------------------------------------------------//

bool
TemporalFieldIO::write(OgOGroup &layerGroup, FieldBase::Ptr field)
{
  using namespace Exc;

  // Add version attribute
  OgOAttribute<int> version(layerGroup, k_versionAttrName, k_versionNumber);

  TemporalField<half>::Ptr halfField = 
    field_dynamic_cast<TemporalField<half> >(field);
  TemporalField<float>::Ptr floatField = 
    field_dynamic_cast<TemporalField<float> >(field);
  TemporalField<double>::Ptr doubleField = 
    field_dynamic_cast<TemporalField<double> >(field);
  TemporalField<V3h>::Ptr vecHalfField = 
    field_dynamic_cast<TemporalField<V3h> >(field);
  TemporalField<V3f>::Ptr vecFloatField = 
    field_dynamic_cast<TemporalField<V3f> >(field);
  TemporalField<V3d>::Ptr vecDoubleField = 
    field_dynamic_cast<TemporalField<V3d> >(field);

  bool success = true;

  if (floatField) {
    success = writeInternal<float>(layerGroup, floatField);
  }
  else if (halfField) {
    success = writeInternal<half>(layerGroup, halfField);
  }
  else if (doubleField) {
    success = writeInternal<double>(layerGroup, doubleField);
  }
  else if (vecFloatField) {
    success = writeInternal<V3f>(layerGroup, vecFloatField);
  }
  else if (vecHalfField) {
    success = writeInternal<V3h>(layerGroup, vecHalfField);
  }
  else if (vecDoubleField) {
    success = writeInternal<V3d>(layerGroup, vecDoubleField);
  }
  else {
    throw WriteLayerException("TemporalFieldIO does not support the given "
                              "TemporalField template parameter");
  }

  return success;
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool TemporalFieldIO::readData(hid_t location, 
                               int numBlocks, 
                               const std::string & /*filename*/, 
                               const std::string & /*layerPath*/, 
                               typename TemporalField<Data_T>::Ptr result)
{
  using namespace std;
  using namespace Exc;
  using namespace Hdf5Util;

  int numOccupiedBlocks;

  int components = FieldTraits<Data_T>::dataDims();

  int offsetsPerBlock = (1 << (result->m_blockOrder * 3)) + 1;
  
  // Read the number of occupied blocks ---

  if (!readAttribute(location, k_numOccupiedBlocksStr, 1, numOccupiedBlocks)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_numOccupiedBlocksStr);

  // Read the block info data sets ---

  TemporalBlock<Data_T> *blocks = result->m_blocks;

  // ... Read the isAllocated array

  vector<char> isAllocated(numBlocks);
  readSimpleData<char>(location, "block_is_allocated_data", isAllocated);

  // Read the data ---

  if (numOccupiedBlocks > 0) {

    // First we need to read the offsets arrays ---

    hsize_t offsetFileDims[1], offsetMemDims[1];
    H5ScopedDopen offsetDataSet(location, k_offsetDataStr, H5P_DEFAULT);
    if (offsetDataSet.id() < 0) {
      throw OpenDataSetException("Couldn't open data set: " + k_offsetDataStr);
    }

    H5ScopedDget_space offsetFileDataSpace(offsetDataSet.id()); 
    if (offsetFileDataSpace.id() < 0) {
      throw GetDataSpaceException("Couldn't get offset data space");
    }
#if 0
    H5ScopedDget_type offsetDataType(offsetDataSet.id());
    if (offsetDataType.id() < 0) {
      throw GetDataTypeException("Couldn't get offset data type");
    }
#endif

    // Offset mem data spaces
    offsetMemDims[0] = offsetsPerBlock;
    H5ScopedScreate offsetMemDataSpace(H5S_SIMPLE);
    H5Sset_extent_simple(offsetMemDataSpace.id(), 1, offsetMemDims, NULL);

    // Offset file dims
    H5Sget_simple_extent_dims(offsetFileDataSpace.id(), offsetFileDims, NULL);
    if (offsetFileDims[0] != 
        static_cast<hsize_t>(offsetsPerBlock * numOccupiedBlocks))  {
      throw FileIntegrityException("Offset data length mismatch in "
                                   "TemporalField");
    }

    // Time and value data sets
    H5ScopedDopen timeDataSet(location, k_timeDataStr, H5P_DEFAULT);
    if (timeDataSet.id() < 0) {
      throw OpenDataSetException("Couldn't open data set: " + k_timeDataStr);
    }
    H5ScopedDopen valueDataSet(location, k_valueDataStr, H5P_DEFAULT);
    if (valueDataSet.id() < 0) {
      throw OpenDataSetException("Couldn't open data set: " + k_valueDataStr);
    }

    // Time and value data spaces
    H5ScopedDget_space timeFileDataSpace(timeDataSet.id()); 
    if (timeFileDataSpace.id() < 0) {
      throw GetDataSpaceException("Couldn't get offset data space");
    }
    H5ScopedDget_space valueFileDataSpace(valueDataSet.id()); 
    if (valueFileDataSpace.id() < 0) {
      throw GetDataSpaceException("Couldn't get offset data space");
    }

    // Time and value file dims
    hsize_t timeFileDims[1], valueFileDims[1];
    H5Sget_simple_extent_dims(timeFileDataSpace.id(), timeFileDims, NULL);
    H5Sget_simple_extent_dims(valueFileDataSpace.id(), valueFileDims, NULL);

    // Read block data one by one ---

    size_t offsetIdx = 0, timeIdx = 0, valueIdx = 0;
    hsize_t offsetOffset[1], offsetCount[1];
    hsize_t timeOffset[1], timeCount[1];
    hsize_t valueOffset[1], valueCount[1];
    herr_t status;

    for (int i = 0; i < numBlocks; ++i) {

      if (isAllocated[i]) {

        // Allocate offset vector. Just 'new' it, the TemporalBlock will
        // assume ownership.
        int *offsetData = new int[offsetsPerBlock];
        // First read the offset array, this will tell us how many samples
        // are in the block
        offsetOffset[0] = offsetIdx;
        offsetCount[0] = offsetsPerBlock;
        status = H5Sselect_hyperslab(offsetFileDataSpace.id(), H5S_SELECT_SET,
                                     offsetOffset, NULL, offsetCount, NULL);
        if (status < 0) {
          throw ReadHyperSlabException("Couldn't select offset slab " + 
                                       boost::lexical_cast<std::string>(i));
        }
        status = H5Dread(offsetDataSet.id(), DataTypeTraits<int>::h5type(), 
                         offsetMemDataSpace.id(), offsetFileDataSpace.id(), 
                         H5P_DEFAULT, offsetData);
        if (status < 0) {
          throw ReadHyperSlabException("Couldn't read offset slab " + 
                                       boost::lexical_cast<std::string>(i));
        }
        // Check num samples
        int numSamples = offsetData[offsetsPerBlock - 1];
        // Allocate time and value arrays. The TemporalBlock will assume 
        // ownership.
        float *timeData = new float[numSamples];
        Data_T *valueData = new Data_T[numSamples * components];
        // Time and value mem data spaces
        hsize_t timeMemDims[1], valueMemDims[1];
        H5ScopedScreate timeMemDataSpace(H5S_SIMPLE);
        timeMemDims[0] = numSamples;
        H5Sset_extent_simple(timeMemDataSpace.id(), 1, timeMemDims, NULL);
        H5ScopedScreate valueMemDataSpace(H5S_SIMPLE);
        valueMemDims[0] = numSamples * components;
        H5Sset_extent_simple(valueMemDataSpace.id(), 1, valueMemDims, NULL);
        // Read the time and value arrays
        timeOffset[0] = timeIdx;
        timeCount[0] = numSamples;
        valueOffset[0] = valueIdx;
        valueCount[0] = numSamples * components;
        status = H5Sselect_hyperslab(timeFileDataSpace.id(), H5S_SELECT_SET,
                                     timeOffset, NULL, timeCount, NULL);
        if (status < 0) {
          throw ReadHyperSlabException("Couldn't select time slab " + 
                                       boost::lexical_cast<std::string>(i));
        }
        status = H5Dread(timeDataSet.id(), DataTypeTraits<float>::h5type(), 
                         timeMemDataSpace.id(), timeFileDataSpace.id(), 
                         H5P_DEFAULT, timeData);
        if (status < 0) {
          throw ReadHyperSlabException("Couldn't read time slab " + 
                                       boost::lexical_cast<std::string>(i));
        }
        status = H5Sselect_hyperslab(valueFileDataSpace.id(), H5S_SELECT_SET,
                                     valueOffset, NULL, valueCount, NULL);
        if (status < 0) {
          throw ReadHyperSlabException("Couldn't select value slab " + 
                                       boost::lexical_cast<std::string>(i));
        }
        status = H5Dread(valueDataSet.id(), DataTypeTraits<Data_T>::h5type(), 
                         valueMemDataSpace.id(), valueFileDataSpace.id(), 
                         H5P_DEFAULT, valueData);
        if (status < 0) {
          throw ReadHyperSlabException("Couldn't read value slab " + 
                                       boost::lexical_cast<std::string>(i));
        }
        // Send arrays to TemporalBlock
        blocks[i].setArrays(offsetData, timeData, valueData);
        // Increment indices
        offsetIdx += offsetCount[0];
        timeIdx += timeCount[0];
        valueIdx += valueCount[0];

      } // isAllocated

    } // for each block

  } // if numOccupiedBlocks > 0

  return true;
  
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename TemporalField<Data_T>::Ptr
TemporalFieldIO::readData(const OgIGroup &location, const Box3i &extents, 
                          const Box3i &dataW, const size_t blockOrder, 
                          const size_t numBlocks, 
                          const std::string &/* filename */, 
                          const std::string &/* layerPath */)
{
  using namespace std;
  using namespace Exc;

  typename TemporalField<Data_T>::Ptr result(new TemporalField<Data_T>);
  result->setSize(extents, dataW);
  result->setBlockOrder(blockOrder);

  const int    numOffsetsPerBlock = (1 << (result->m_blockOrder * 3)) + 1;
  // const size_t numVoxelsPerBlock  = (1 << (blockOrder * 3));

  // Read the number of occupied blocks ---

  OgIAttribute<uint32_t> occupiedBlocksAttr = 
    location.findAttribute<uint32_t>(k_numOccupiedBlocksStr);
  if (!occupiedBlocksAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_numOccupiedBlocksStr);
  }
  const size_t occupiedBlocks = occupiedBlocksAttr.value();

  // Read the block info data sets ---

  TemporalBlock<Data_T> *blocks = result->m_blocks;

  // ... Read the isAllocated array

  vector<uint8_t> isAllocated(numBlocks);

  {
    // Grab the data
    OgIDataset<uint8_t> isAllocatedData = 
      location.findDataset<uint8_t>("block_is_allocated_data");
    if (!isAllocatedData.isValid()) {
      throw MissingGroupException("Couldn't find block_is_allocated_data: ");
    }
    isAllocatedData.getData(0, &isAllocated[0], OGAWA_THREAD);
  }

  // Read the data ---

  // Compressed data caches
  std::vector<uint8_t> offsetCache, timeCache, valueCache;

  // Find the data sets
  OgICDataset<int32_t> offsetDataset = 
    location.findCompressedDataset<int32_t>(k_offsetDataStr);
  OgICDataset<float32_t> timeDataset = 
    location.findCompressedDataset<float32_t>(k_timeDataStr);
  OgICDataset<Data_T> valueDataset = 
    location.findCompressedDataset<Data_T>(k_valueDataStr);
      
  // Check validity
  if (!offsetDataset.isValid()) {
    throw ReadDataException("Couldn't open compressed data set: " + 
                            k_offsetDataStr);
  }
  if (!timeDataset.isValid()) {
    throw ReadDataException("Couldn't open compressed data set: " + 
                            k_timeDataStr);
  }
  if (!valueDataset.isValid()) {
    throw ReadDataException("Couldn't open compressed data set: " + 
                            k_valueDataStr);
  }
  if (offsetDataset.numDataElements() != occupiedBlocks ||
      timeDataset.numDataElements() != occupiedBlocks ||
      valueDataset.numDataElements() != occupiedBlocks) {
    throw ReadDataException("Block count mismatch in TemporalFieldIO");
  }

  // Check data type of values
  OgDataType typeOnDisk = location.compressedDatasetType(k_valueDataStr);
  if (typeOnDisk != OgawaTypeTraits<Data_T>::typeEnum()) {
    throw ReadDataException("Data type mismatch in TemporalFieldIO");
  }

  // Set compression cache size
  offsetCache.resize(compressBound(numOffsetsPerBlock * sizeof(int32_t)));
      
  // Data set index. Note that this may be smaller than numBlocks
  size_t curIdx = 0;

  for (size_t i = 0; i < numBlocks; ++i) {
    if (isAllocated[i]) {
      // Allocate offset storage. Just 'new' it, the TemporalBlock will
      // assume ownership.
      int *offsetData   = new int[numOffsetsPerBlock];
      // Read offset data
      offsetDataset.getData(curIdx, &offsetCache[0], OGAWA_THREAD);
      // Decompress the offset data
      uint64_t  length   = offsetDataset.dataSize(curIdx, OGAWA_THREAD);
      uint8_t  *ucmpData = reinterpret_cast<uint8_t *>(offsetData);
      uLong     ucmpLen  = numOffsetsPerBlock * sizeof(int32_t);
      int       status   = uncompress(ucmpData, &ucmpLen, 
                                      &offsetCache[0], length);
      if (status != Z_OK) {
        std::cout << "ERROR in uncompress offsets: " << status
                  << " " << ucmpLen << " " << length << std::endl;
        throw ReadDataException("Decompression error in TemporalFieldIO");
      }
      // Compute number of samples
      const int numSamples = offsetData[numOffsetsPerBlock - 1];
      // Allocate sample data
      float  *timeData  = new float[numSamples];
      Data_T *valueData = new Data_T[numSamples];
      // Resize caches
      timeCache.resize(compressBound(numSamples * sizeof(float32_t)));
      valueCache.resize(compressBound(numSamples * sizeof(Data_T)));
      // Read sample data
      timeDataset.getData(curIdx, &timeCache[0], OGAWA_THREAD);
      valueDataset.getData(curIdx, &valueCache[0], OGAWA_THREAD);
      // Decompress sample data
      length   = timeDataset.dataSize(curIdx, OGAWA_THREAD);
      ucmpData = reinterpret_cast<uint8_t *>(timeData);
      ucmpLen  = numSamples * sizeof(float32_t);
      status   = uncompress(ucmpData, &ucmpLen, &timeCache[0], length);
      if (status != Z_OK) {
        std::cout << "ERROR in uncompress time: " << status
                  << " " << ucmpLen << " " << length << std::endl;
        throw ReadDataException("Decompression error in TemporalFieldIO");
      }
      length   = valueDataset.dataSize(curIdx, OGAWA_THREAD);
      ucmpData = reinterpret_cast<uint8_t *>(valueData);
      ucmpLen  = numSamples * sizeof(float32_t);
      status   = uncompress(ucmpData, &ucmpLen, &valueCache[0], length);
      if (status != Z_OK) {
        std::cout << "ERROR in uncompress value: " << status
                  << " " << ucmpLen << " " << length << std::endl;
        throw ReadDataException("Decompression error in TemporalFieldIO");
      }
      // Send arrays to TemporalBlock
      blocks[i].setArrays(offsetData, timeData, valueData);
      // Step to next index
      curIdx++;
    }
  }

  return result;
}

//----------------------------------------------------------------------------//
// Template methods
//----------------------------------------------------------------------------//

//! \todo Tune the chunk size of the gzip call
template <class Data_T>
bool TemporalFieldIO::writeInternal(hid_t layerGroup, 
                                    typename TemporalField<Data_T>::Ptr field)
{
  using namespace std;
  using namespace Exc;
  using namespace Hdf5Util;

  typedef TemporalBlock<Data_T> Block;

  Box3i ext(field->extents()), dw(field->dataWindow());

  int components = FieldTraits<Data_T>::dataDims();

  int offsetsPerBlock = (1 << (field->m_blockOrder * 3)) + 1;

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
  
  Block *blocks = field->m_blocks;

  // Write the isAllocated array
  vector<char> isAllocated(numBlocks);
  {
    for (int i = 0; i < numBlocks; ++i) {
      isAllocated[i] = 
        static_cast<char>(blocks[i].state() == Block::ActiveState);
    }
    writeSimpleData<char>(layerGroup, "block_is_allocated_data", isAllocated);
  }

  // Count the number of occupied blocks ---

  int numOccupiedBlocks = 0;
  int numTotalSamples = 0;
  for (int i = 0; i < numBlocks; ++i) {
    if (blocks[i].state() == TemporalBlock<Data_T>::ActiveState) {
      numOccupiedBlocks++;
      numTotalSamples += blocks[i].numSamples(field->m_blockOrder);
    }
  }

  if (!writeAttribute(layerGroup, k_numOccupiedBlocksStr, 1, numOccupiedBlocks)) {
    throw WriteAttributeException("Couldn't add attribute " + 
                                  k_numOccupiedBlocksStr);
  }
  
  if (numOccupiedBlocks > 0) {

    // Set up gzip property list
    bool gzipAvailable = checkHdf5Gzip();
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunkSize[1];
    chunkSize[0] = std::min(4096, numTotalSamples);
    if (gzipAvailable) {
      herr_t status = H5Pset_deflate(dcpl, 9);
      if (status < 0) {
        Msg::print(Msg::SevWarning, "Couldn't set gzip deflate.");
        return false;
      }
      status = H5Pset_chunk(dcpl, 1, chunkSize);
      if (status < 0) {
        Msg::print(Msg::SevWarning, "Couldn't set gzip chunk.");
        return false;
      }    
    }

    // Make the offset, time, and value dimensions
    hsize_t offsetMemDims[1], timeMemDims[1], valueMemDims[1];
    hsize_t offsetFileDims[1], timeFileDims[1], valueFileDims[1];
    offsetMemDims[0] = offsetsPerBlock;
    offsetFileDims[0] = offsetsPerBlock * numOccupiedBlocks;
    timeFileDims[0] = numTotalSamples;
    valueFileDims[0] = numTotalSamples * components;

    // Make the data spaces
    H5ScopedScreate offsetMemDataSpace(H5S_SIMPLE);
    H5ScopedScreate offsetFileDataSpace(H5S_SIMPLE);
    H5ScopedScreate timeFileDataSpace(H5S_SIMPLE);
    H5ScopedScreate valueFileDataSpace(H5S_SIMPLE);
    H5Sset_extent_simple(offsetMemDataSpace.id(), 1, offsetMemDims, NULL);
    H5Sset_extent_simple(offsetFileDataSpace.id(), 1, offsetFileDims, NULL);
    H5Sset_extent_simple(timeFileDataSpace.id(), 1, timeFileDims, NULL);
    H5Sset_extent_simple(valueFileDataSpace.id(), 1, valueFileDims, NULL);
    
    // Add the data sets
    H5ScopedDcreate offsetDataSet(layerGroup, k_offsetDataStr, 
                                  DataTypeTraits<int>::h5type(), 
                                  offsetFileDataSpace.id(), 
                                  H5P_DEFAULT, dcpl, H5P_DEFAULT);

    if (offsetDataSet.id() < 0) {
      throw CreateDataSetException("Couldn't create offset data set in "
                                   "TemporalFieldIO::writeInternal");
    }
    
    H5ScopedDcreate timeDataSet(layerGroup, k_timeDataStr, 
                                DataTypeTraits<float>::h5type(), 
                                timeFileDataSpace.id(), 
                                H5P_DEFAULT, dcpl, H5P_DEFAULT);
    
    if (timeDataSet.id() < 0) {
      throw CreateDataSetException("Couldn't create time data set in "
                                   "TemporalFieldIO::writeInternal");
    }
    
    H5ScopedDcreate valueDataSet(layerGroup, k_valueDataStr, 
                                 DataTypeTraits<Data_T>::h5type(), 
                                 valueFileDataSpace.id(), 
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
    
    if (valueDataSet.id() < 0) {
      throw CreateDataSetException("Couldn't create value data set in "
                                   "TemporalFieldIO::writeInternal");
    }
    
    // For each block, we have a new time and value memory dimension
    size_t  offsetIdx = 0, timeIdx = 0, valueIdx = 0;
    hsize_t offsetOffset[1], timeOffset[1], valueOffset[1];
    hsize_t offsetCount[1], timeCount[1], valueCount[1];
    herr_t  status;

    for (int i = 0; i < numBlocks; ++i) {

      if (!isAllocated[i]) {
        continue;
      }
      
      size_t numSamples  = blocks[i].numSamples(field->m_blockOrder);

      timeMemDims[0]     = numSamples;
      valueMemDims[0]    = numSamples;

      offsetOffset[0]    = offsetIdx;
      offsetCount[0]     = offsetsPerBlock;

      timeOffset[0]      = timeIdx;
      timeCount[0]       = numSamples;

      valueOffset[0]     = valueIdx;
      valueCount[0]      = numSamples * components;

      // Time and value data space
      H5ScopedScreate timeMemDataSpace(H5S_SIMPLE);
      H5ScopedScreate valueMemDataSpace(H5S_SIMPLE);
      timeMemDims[0] = numSamples;
      valueMemDims[0] = numSamples * components;
      H5Sset_extent_simple(timeMemDataSpace.id(), 1, timeMemDims, NULL);
      H5Sset_extent_simple(valueMemDataSpace.id(), 1, valueMemDims, NULL);
      
      // Select hyperslabs
      status = H5Sselect_hyperslab(offsetFileDataSpace.id(), H5S_SELECT_SET, 
                                   offsetOffset, NULL, offsetCount, NULL);
      if (status < 0) {
        throw WriteHyperSlabException("Couldn't select offset slab " + 
                                      boost::lexical_cast<std::string>(i));
      }
      status = H5Sselect_hyperslab(timeFileDataSpace.id(), H5S_SELECT_SET, 
                                   timeOffset, NULL, timeCount, NULL);
      if (status < 0) {
        throw WriteHyperSlabException("Couldn't select time slab " + 
                                      boost::lexical_cast<std::string>(i));
      }
      status = H5Sselect_hyperslab(valueFileDataSpace.id(), H5S_SELECT_SET, 
                                   valueOffset, NULL, valueCount, NULL);
      if (status < 0) {
        throw WriteHyperSlabException("Couldn't select value slab " + 
                                      boost::lexical_cast<std::string>(i));
      }

      // Write data
      const int *offsetData = field->m_blocks[i].offsets();
      status = H5Dwrite(offsetDataSet.id(), DataTypeTraits<int>::h5type(), 
                        offsetMemDataSpace.id(), offsetFileDataSpace.id(), 
                        H5P_DEFAULT, offsetData);
      const float *timeData = field->m_blocks[i].times();
      status = H5Dwrite(timeDataSet.id(), DataTypeTraits<float>::h5type(), 
                        timeMemDataSpace.id(), timeFileDataSpace.id(), 
                        H5P_DEFAULT, timeData);
      const Data_T *valueData = field->m_blocks[i].values();
      status = H5Dwrite(valueDataSet.id(), DataTypeTraits<Data_T>::h5type(), 
                        valueMemDataSpace.id(), valueFileDataSpace.id(), 
                        H5P_DEFAULT, valueData);
      
      // Go to next offset
      offsetIdx += offsetCount[0];
      timeIdx += timeCount[0];
      valueIdx += valueCount[0];

    }

  } // if numOccupiedBlocks > 0

  return true; 

}

//----------------------------------------------------------------------------//

namespace {

template <typename Data_T>
struct ThreadingState
{
#if 1
  ThreadingState(OgOCDataset<int32_t>        &i_offsetData, 
                 OgOCDataset<float32_t>      &i_timeData, 
                 OgOCDataset<Data_T>         &i_valueData, 
                 TemporalBlock<Data_T>       *i_blocks, 
                 const size_t                 i_numVoxels, 
                 const size_t                 i_numBlocks,
                 const size_t                 i_blockOrder,
                 const std::vector<uint8_t>  &i_isAllocated)
    : offsetData(i_offsetData),
      timeData(i_timeData),
      valueData(i_valueData), 
      blocks(i_blocks),
      numVoxels(i_numVoxels), 
      numBlocks(i_numBlocks),
      blockOrder(i_blockOrder), 
      isAllocated(i_isAllocated), 
      nextBlockToCompress(0),
      nextBlockToWrite(0)
  { 
    // Find first in-use block
    for (size_t i = 0; i < numBlocks; ++i) {
      if (blocks[i].state() == TemporalBlock<Data_T>::ActiveState) {
        nextBlockToCompress = i;
        nextBlockToWrite = i;
        return;
      }
    }
    // If we get here, there are no active blocks. Set to numBlocks
    nextBlockToCompress = numBlocks;
    nextBlockToWrite = numBlocks;
  }
  // Data members
  OgOCDataset<int32_t>        &offsetData;
  OgOCDataset<float32_t>      &timeData;
  OgOCDataset<Data_T>         &valueData;
  TemporalBlock<Data_T>       *blocks;
  const size_t                 numVoxels;
  const size_t                 numBlocks;
  const size_t                 blockOrder;
  const std::vector<uint8_t>   isAllocated;
  size_t                       nextBlockToCompress;
  size_t                       nextBlockToWrite;
  // Mutexes
  boost::mutex compressMutex;
#endif
};

//----------------------------------------------------------------------------//

template <typename T>
bool compress(const T *data, const size_t numValues, 
              std::vector<uint8_t> &cache, uLong &cmpLen)
{
  const int level = 1;
  // Block data as bytes
  const uint8_t *srcData = reinterpret_cast<const uint8_t *>(data);
  // Length of compressed data is stored here
  const uLong srcLen      = numValues * sizeof(T);
  const uLong cmpLenBound = compressBound(srcLen);
  // Initialize the compressed length
  cmpLen = cmpLenBound;
  // Make sure the cache is large enough
  if (cache.size() < cmpLenBound) {
    cache.resize(cmpLenBound);
  }
  // Perform compression
  const int status = compress2(&cache[0], &cmpLen, srcData, srcLen, level);
  // Error check
  if (status != Z_OK) {
    std::cout << "ERROR: Couldn't compress in TemporalFieldIO." << std::endl
              << "  Level:  " << level << std::endl
              << "  Status: " << status << std::endl
              << "  srcLen: " << srcLen << std::endl
              << "  cmpLenBound: " << cmpLenBound << std::endl
              << "  cmpLen: " << cmpLen << std::endl;
    return false;
  }
  // Done
  return true;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
class WriteBlockOp
{
public:
  WriteBlockOp(ThreadingState<Data_T> &state)
    : m_state(state)
  { }
  void operator() ()
  {
    std::vector<uint8_t> offsetCache, timeCache, valueCache;
    const int offsetsPerBlock = m_state.numVoxels + 1;
    // Get next block to compress
    size_t blockIdx;
    {
      boost::mutex::scoped_lock lock(m_state.compressMutex);
      blockIdx = m_state.nextBlockToCompress;
      // Step counter to next
      while (m_state.nextBlockToCompress < m_state.numBlocks) {
        m_state.nextBlockToCompress++;
        if (m_state.blocks[m_state.nextBlockToCompress].state() ==
            TemporalBlock<Data_T>::ActiveState) {
          break;
        }
      }
    }
    // Loop over blocks until we run out
    while (blockIdx < m_state.numBlocks) {
      if (m_state.blocks[blockIdx].state() == 
          TemporalBlock<Data_T>::ActiveState) {
        const size_t numSamples  = 
          m_state.blocks[blockIdx].numSamples(m_state.blockOrder);
        uLong offsetLen = 0, timeLen = 0, valueLen = 0;
        compress(m_state.blocks[blockIdx].offsets(), offsetsPerBlock,
                 offsetCache, offsetLen);
        compress(m_state.blocks[blockIdx].times(), numSamples,
                 timeCache, timeLen);
        compress(m_state.blocks[blockIdx].values(), numSamples,
                 valueCache, valueLen);
        // Wait to write data
        while (m_state.nextBlockToWrite != blockIdx) {
          // Spin
          boost::this_thread::sleep(boost::posix_time::microseconds(1));
        }
        // Do the writing
        m_state.offsetData.addData(offsetLen, &offsetCache[0]);
        m_state.timeData.addData(timeLen, &timeCache[0]);
        m_state.valueData.addData(valueLen, &valueCache[0]);
        // Let next block write
        while (m_state.nextBlockToWrite < m_state.numBlocks){
          // Increment to next
          m_state.nextBlockToWrite++;
          if (m_state.blocks[m_state.nextBlockToWrite].state() ==
              TemporalBlock<Data_T>::ActiveState) {
            break;
          }
        }
      }
      // Get next block idx
      {
        boost::mutex::scoped_lock lock(m_state.compressMutex);
        blockIdx = m_state.nextBlockToCompress;
        // Step counter to next
        while (m_state.nextBlockToCompress < m_state.numBlocks) {
          m_state.nextBlockToCompress++;
          if (m_state.blocks[m_state.nextBlockToCompress].state() ==
              TemporalBlock<Data_T>::ActiveState) {
            break;
          }
        }
      }
    }
  }
private:
  // Data members ---
  ThreadingState<Data_T> &m_state;
};

}

//----------------------------------------------------------------------------//

template <class Data_T>
bool TemporalFieldIO::writeInternal(OgOGroup &layerGroup, 
                                  typename TemporalField<Data_T>::Ptr field)
{
  using namespace Exc;

  typedef TemporalBlock<Data_T> Block;

  Block *blocks = field->m_blocks;

  const int     components      = FieldTraits<Data_T>::dataDims();
  const int     bits            = DataTypeTraits<Data_T>::h5bits();
  const V3i    &blockRes        = field->m_blockRes;
  const size_t  numBlocks       = blockRes.x * blockRes.y * blockRes.z;
  const size_t  numVoxels       = (1 << (field->m_blockOrder * 3));

  const Box3i ext(field->extents()), dw(field->dataWindow());

  // Add attributes ---

  OgOAttribute<veci32_t> extMinAttr(layerGroup, k_extentsMinStr, ext.min);
  OgOAttribute<veci32_t> extMaxAttr(layerGroup, k_extentsMaxStr, ext.max);
  
  OgOAttribute<veci32_t> dwMinAttr(layerGroup, k_dataWindowMinStr, dw.min);
  OgOAttribute<veci32_t> dwMaxAttr(layerGroup, k_dataWindowMaxStr, dw.max);

  OgOAttribute<uint8_t> componentsAttr(layerGroup, k_componentsStr, components);

  OgOAttribute<uint8_t> bitsAttr(layerGroup, k_bitsPerComponentStr, bits);

  OgOAttribute<uint8_t> blockOrderAttr(layerGroup, k_blockOrderStr, 
                                       field->m_blockOrder);

  OgOAttribute<uint32_t> numBlocksAttr(layerGroup, k_numBlocksStr, numBlocks);

  OgOAttribute<veci32_t> blockResAttr(layerGroup, k_blockResStr, blockRes);

  // Write the isAllocated array
  std::vector<uint8_t> isAllocated(numBlocks);
  for (size_t i = 0; i < numBlocks; ++i) {
    isAllocated[i] = 
      static_cast<uint8_t>(blocks[i].state() == Block::ActiveState);
  }
  OgODataset<uint8_t> isAllocatedData(layerGroup, "block_is_allocated_data");
  isAllocatedData.addData(numBlocks, &isAllocated[0]);

  // Count the number of occupied blocks ---

  int numOccupiedBlocks = 0;
  int numTotalSamples = 0;
  for (size_t i = 0; i < numBlocks; ++i) {
    if (blocks[i].state() == TemporalBlock<Data_T>::ActiveState) {
      numOccupiedBlocks++;
      numTotalSamples += blocks[i].numSamples(field->m_blockOrder);
    }
  }

  OgOAttribute<uint32_t> numOccupiedBlockAttr(layerGroup, 
                                              k_numOccupiedBlocksStr, 
                                              numOccupiedBlocks);

  // Add data to file ---

  // Create the compressed dataset regardless of whether there are blocks
  // to write.
  OgOCDataset<int32_t> offsetData(layerGroup, k_offsetDataStr);
  OgOCDataset<float32_t> timeData(layerGroup, k_timeDataStr);
  OgOCDataset<Data_T> valueData(layerGroup, k_valueDataStr);
  // Write data if there is any
  if (numOccupiedBlocks > 0) {
    // Threading state
    ThreadingState<Data_T> state(offsetData, timeData, valueData, 
                                 blocks, numVoxels, numBlocks, 
                                 field->m_blockOrder, isAllocated);
    // Number of threads
    const size_t numThreads = numIOThreads();
    // Launch threads
    boost::thread_group threads;
    for (size_t i = 0; i < numThreads; ++i) {
      threads.create_thread(WriteBlockOp<Data_T>(state));
    }
    threads.join_all();
  }

  return true;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//

