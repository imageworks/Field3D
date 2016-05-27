//----------------------------------------------------------------------------//
// Includes
//----------------------------------------------------------------------------//

#include <iostream>

#include "OgawaUtil.h"

//----------------------------------------------------------------------------//

using std::cout;
using std::endl;

using namespace Field3D;

//----------------------------------------------------------------------------//
// Globals
//----------------------------------------------------------------------------//

int g_indent = 0;
bool g_printDatasets = false;
bool g_printDatasetInfo = true;
bool g_printDatasetDetails = false;

//----------------------------------------------------------------------------//
// Function prototypes
//----------------------------------------------------------------------------//

void ogawaInfo(const std::string &filename);

void printHierarchy(OgIGroup group);

//----------------------------------------------------------------------------//
// main()
//----------------------------------------------------------------------------//

int main(int argc, const char *argv[])
{
  if (argc < 2) {
    return 0;
  }

  const char *filename = argv[1];

  if (argc > 2) {
    std::string arg2 = argv[2];
    g_printDatasetInfo = (arg2 == "-d");
    g_printDatasetDetails = (arg2 == "-da");
    g_printDatasets = g_printDatasetDetails || g_printDatasetInfo;
  }

  ogawaInfo(filename);
}

//----------------------------------------------------------------------------//
// Function implementations
//----------------------------------------------------------------------------//

std::string inQuotes(const std::string &s)
{
  return "\"" + s + "\"";
}

//----------------------------------------------------------------------------//

void indent()
{
  for (int i = 0; i < g_indent; ++i) {
    cout << "  ";
  }
}

//----------------------------------------------------------------------------//

void ogawaInfo(const std::string &filename)
{
  using namespace Alembic::Ogawa;

  IArchive ia(filename);
  if (!ia.isValid()) {
    cout << "Couldn't open file: " << filename << endl;
  }

  cout << "F3D-Ogawa contents : " << filename << endl << endl;

  OgIGroup root(ia);
  printHierarchy(root);
}

//----------------------------------------------------------------------------//

int printable(const uint8_t &value)
{
  return static_cast<int>(value);
}

//----------------------------------------------------------------------------//

template <typename T>
const T& printable(const T &value)
{
  return value;
}

//----------------------------------------------------------------------------//

template <typename T>
void printAttr(const OgIAttribute<T> &attr)
{
  // Print, but only if valid
  if (attr.isValid()) {
    indent();
    cout << "  a " << inQuotes(attr.name()) << " : " 
         << printable(attr.value()) 
         << " (" << OgawaTypeTraits<T>::typeName() << ")" << endl;
  } 
}

//----------------------------------------------------------------------------//

template <typename T>
void printDataset(const OgIDataset<T> &dataset)
{
  // Print, but only if valid
  if (dataset.isValid()) {
    g_indent++;
    indent();
    cout << "d " << inQuotes(dataset.name()) << " (" 
         << OgawaTypeTraits<T>::typeName() << ")" << endl;
    indent();
    cout << "  num elements: " << dataset.numDataElements() << endl;
    // Print data set lengths
    size_t totalElems = 0;
    if (g_printDatasets) {
      std::vector<T> tempData;
      for (int i = 0, end = dataset.numDataElements(); i < end; ++i) {
        const size_t dataSize = dataset.dataSize(i, 0);
        totalElems += dataSize;
        indent();
        if (g_printDatasetInfo) {
          // Header
          cout << "    " << i << " : " << dataset.dataSize(i, 0) << endl;
        } else if (g_printDatasetDetails) {
          // Header
          cout << "    " << i << " (" << dataset.dataSize(i, 0) << ")" << endl;
          if (g_printDatasetDetails) {
            // Resize storage
            if (tempData.size() < dataSize) {
              tempData.resize(dataSize);
            }
            // Get data
            dataset.getData(i, &tempData[0], 0);
            // Print data
            for (size_t e = 0; e < dataSize; ++e) {
              indent();
              cout << "      " << e << " : " << printable(tempData[e]) << endl;
            }
          }
        }
      }      
    } 
    // Print size
    if (totalElems > 0) {
      indent();
      cout << "  total length: " << totalElems << endl;
    }
    // Done
    g_indent--;
  }
}

//----------------------------------------------------------------------------//

template <typename T>
void printDataset(const OgICDataset<T> &dataset)
{
  // Print, but only if valid
  if (dataset.isValid()) {
    g_indent++;
    indent();
    cout << "d " << inQuotes(dataset.name()) << " (compressed " 
         << OgawaTypeTraits<T>::typeName() << ")" << endl;
    indent();
    cout << "  num elements: " << dataset.numDataElements() << endl;
    // Print data set lengths
    size_t totalElems = 0;
    if (g_printDatasets) {
      std::vector<T> tempData;
      for (int i = 0, end = dataset.numDataElements(); i < end; ++i) {
        const size_t dataSize = dataset.dataSize(i, 0);
        totalElems += dataSize;
        indent();
        if (g_printDatasetInfo) {
          // Header
          cout << "    " << i << " : " << dataset.dataSize(i, 0) << endl;
        } else if (g_printDatasetDetails) {
#if 0
          // Header
          cout << "    " << i << " (" << dataset.dataSize(i, 0) << ")" << endl;
          if (g_printDatasetDetails) {
            // Resize storage
            if (tempData.size() < dataSize) {
              tempData.resize(dataSize);
            }
            // Get data
            dataset.getData(i, &tempData[0], 0);
            // Print data
            for (size_t e = 0; e < dataSize; ++e) {
              indent();
              cout << "      " << e << " : " << printable(tempData[e]) << endl;
            }
          }
#endif
        }
      }      
    } 
    // Print size
    if (totalElems > 0) {
      indent();
      cout << "  total length: " << totalElems << endl;
    }
    // Done
    g_indent--;
  }
}

//----------------------------------------------------------------------------//

#define PRINT_ATTR(type)                                            \
  {                                                                 \
    OgIAttribute<type> attr = group.findAttribute<type>(attrs[i]);  \
    printAttr(attr);                                                \
  }

#define PRINT_DATASET(type)                                             \
  {                                                                     \
    OgIDataset<type> dataset = group.findDataset<type>(datasets[i]);    \
    printDataset(dataset);                                              \
  }

#define PRINT_COMPRESSED_DATASET(type)                                  \
  {                                                                     \
    OgICDataset<type> dataset =                                         \
      group.findCompressedDataset<type>(datasets[i]);                   \
    printDataset(dataset);                                              \
  }

void printHierarchy(const OgIGroup group)
{
  // Print self
  indent();
  cout << "g " << inQuotes(group.name()) << endl;
  // Print attributes
  std::vector<std::string> attrs = group.attributeNames();
  for (size_t i = 0, end = attrs.size(); i < end; ++i) {
    PRINT_ATTR(int8_t);
    PRINT_ATTR(uint8_t);
    PRINT_ATTR(int16_t);
    PRINT_ATTR(uint16_t);
    PRINT_ATTR(int32_t);
    PRINT_ATTR(uint32_t);
    PRINT_ATTR(int64_t);
    PRINT_ATTR(uint64_t);
    PRINT_ATTR(float16_t);
    PRINT_ATTR(float32_t);
    PRINT_ATTR(float64_t);
    PRINT_ATTR(vec16_t);
    PRINT_ATTR(vec32_t);
    PRINT_ATTR(vec64_t);
    PRINT_ATTR(veci32_t);
    PRINT_ATTR(std::string);
  }
  // Print datasets
  std::vector<std::string> datasets = group.datasetNames();
  for (size_t i = 0, end = datasets.size(); i < end; ++i) {
    PRINT_DATASET(int8_t);
    PRINT_DATASET(uint8_t);
    PRINT_DATASET(int16_t);
    PRINT_DATASET(uint16_t);
    PRINT_DATASET(int32_t);
    PRINT_DATASET(uint32_t);
    PRINT_DATASET(int64_t);
    PRINT_DATASET(uint64_t);
    PRINT_DATASET(float16_t);
    PRINT_DATASET(float32_t);
    PRINT_DATASET(float64_t);
    PRINT_DATASET(vec16_t);
    PRINT_DATASET(vec32_t);
    PRINT_DATASET(vec64_t);
    PRINT_DATASET(veci32_t);
  }
  // Print datasets
  datasets = group.compressedDatasetNames();
  for (size_t i = 0, end = datasets.size(); i < end; ++i) {
    PRINT_COMPRESSED_DATASET(int8_t);
    PRINT_COMPRESSED_DATASET(uint8_t);
    PRINT_COMPRESSED_DATASET(int16_t);
    PRINT_COMPRESSED_DATASET(uint16_t);
    PRINT_COMPRESSED_DATASET(int32_t);
    PRINT_COMPRESSED_DATASET(uint32_t);
    PRINT_COMPRESSED_DATASET(int64_t);
    PRINT_COMPRESSED_DATASET(uint64_t);
    PRINT_COMPRESSED_DATASET(float16_t);
    PRINT_COMPRESSED_DATASET(float32_t);
    PRINT_COMPRESSED_DATASET(float64_t);
    PRINT_COMPRESSED_DATASET(vec16_t);
    PRINT_COMPRESSED_DATASET(vec32_t);
    PRINT_COMPRESSED_DATASET(vec64_t);
    PRINT_COMPRESSED_DATASET(veci32_t);
  }
  // Check child groups
  std::vector<std::string> groups = group.groupNames();
  for (size_t i = 0, end = groups.size(); i < end; ++i) {
    // Open the group
    OgIGroup childGroup = group.findGroup(groups[i]);
    // Print, if valid
    if (childGroup.isValid()) {
      g_indent++;
      printHierarchy(childGroup);
      g_indent--;
    }
  }
}

//----------------------------------------------------------------------------//
