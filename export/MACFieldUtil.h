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

/*! \file MACFieldUtil.h
  \brief Contains utility functions for MAC fields, such as conversion
  to cell-centered fields.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_MACFieldUtil_H_
#define _INCLUDED_Field3D_MACFieldUtil_H_

#include "MACField.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Utility functions
//----------------------------------------------------------------------------//

//! Converts the MAC field to a cell-centered field
template <class Data_T, class Field_T>
void convertMACToCellCentered(typename MACField<Data_T>::Ptr mac,
                              typename Field_T::Ptr cc);

//----------------------------------------------------------------------------//

//! Converts the cell-centered field to a MAC field
template <class Field_T, class Data_T>
void convertCellCenteredToMAC(typename Field_T::Ptr cc,
                              typename MACField<Data_T>::Ptr mac);

//----------------------------------------------------------------------------//
// Implementations
//----------------------------------------------------------------------------//

//! Sets up the cell-centered target field given a MAC field
template <class Data_T, class Field_T>
void convertMACToCellCentered(typename MACField<Data_T>::Ptr mac,
                              typename Field_T::Ptr cc)
{
  // Make sure the extents and data window match
  if (cc->extents().min != mac->extents().min ||
      cc->extents().max != mac->extents().max ||
      cc->dataWindow().min != mac->dataWindow().min ||
      cc->dataWindow().max != mac->dataWindow().max ) {
    cc->setSize(mac->extents(), mac->dataWindow());
  }

  // Make sure mapping matches
  if (!cc->mapping()->isIdentical(mac->mapping())) {
    cc->setMapping(mac->mapping());
  }

  // MAC velocities are in simulation space (axis-aligned to the
  // mapping) because the values are stored on the faces, so rotate
  // vectors from simulation-space to world-space when transferring
  // from MAC to cell-centered

  bool rotateVector = false;
  M44d ssToWsMtx;
  MatrixFieldMapping::Ptr mapping =
    boost::dynamic_pointer_cast<MatrixFieldMapping>(mac->mapping());
  if (mapping) {
    M44d localToWorldMtx = mapping->localToWorld();
    V3d scale, rot, trans, shear;
    if (extractSHRT(localToWorldMtx, scale, shear, rot, trans, false)) {
      ssToWsMtx.rotate(rot);
      if (rot.length2() > FLT_EPSILON)
        rotateVector = true;
    }
  }

  // Loop over all the voxels in the output field ---

  typename Field_T::iterator i = cc->begin();
  typename Field_T::iterator end = cc->end();

  if (rotateVector) {
    for (; i != end; ++i) {
      *i = mac->value(i.x, i.y, i.z) * ssToWsMtx;
    }
  } else {
    for (; i != end; ++i) {
      *i = mac->value(i.x, i.y, i.z);
    }
  }
}

//----------------------------------------------------------------------------//

template <class Field_T, class Data_T>
void convertCellCenteredToMAC(typename Field_T::Ptr cc,
                              typename MACField<Data_T>::Ptr mac)
{
  // Make sure the extents and data window match
  if (mac->extents().min != cc->extents().min ||
      mac->extents().max != cc->extents().max ||
      mac->dataWindow().min != cc->dataWindow().min ||
      mac->dataWindow().max != cc->dataWindow().max ) {
    mac->setSize(cc->extents(), cc->dataWindow());
  }

  // Make sure mapping matches
  if (!mac->mapping()->isIdentical(cc->mapping())) {
    mac->setMapping(cc->mapping());
  }
  
  Box3i data = mac->dataWindow();

  // MAC velocities are in simulation space (axis-aligned to the
  // mapping) because the values are stored on the faces, so rotate
  // vectors from world-space to simulation-space when transferring
  // from cell-centered to MAC

  bool rotateVector = false;
  M44d wsToSsMtx;
  MatrixFieldMapping::Ptr mapping =
    boost::dynamic_pointer_cast<MatrixFieldMapping>(mac->mapping());
  if (mapping) {
    M44d localToWorld = mapping->localToWorld();
    V3d scale, rot, trans, shear;
    if (FIELD3D_EXTRACT_SHRT(localToWorld, scale, shear, rot, trans, false)) {
      wsToSsMtx.rotate(-rot);
      rotateVector = true;
    }
  }

  // Use a pointer to a field below so we can substitute it out for
  // our intermediate, rotated field if necessary, without needing to
  // duplicate the loops.  This should be more efficient CPU-wise so
  // we don't need to do 3 matrix multiplies per cell-centered voxel
  // because it's used in 3 separate loops (1 per MAC face).
  typename Field_T::Ptr src = cc;

  typename Field_T::Ptr ccRotated;
  if (rotateVector) {
    ccRotated =
      typename Field_T::Ptr(new Field_T);
    ccRotated->matchDefinition(cc);

    typename Field_T::const_iterator iIn = cc->cbegin();
    typename Field_T::const_iterator endIn = cc->cend();
    typename Field_T::iterator iOut = ccRotated->begin();

    for (; iIn != endIn; ++iIn, ++iOut) {
      *iOut = *iIn * wsToSsMtx;
    }
    src = ccRotated;
  }

  // Set the u edge value to their closest voxel
  for (int k = data.min.z; k <= data.max.z; k++) {
    for (int j = data.min.y; j <= data.max.y; j++) {
      mac->u(data.min.x, j, k) = src->value(data.min.x, j, k).x;
      mac->u(data.max.x + 1, j, k) = src->value(data.max.x, j, k).x;
    }
  }

  // Set the v edge value to their closest voxel
  for (int k = data.min.z; k <= data.max.z; k++) {
    for (int i = data.min.x; i <= data.max.x; i++) {
      mac->v(i, data.min.y, k) = src->value(i, data.min.y, k).y;
      mac->v(i, data.max.y + 1, k) = src->value(i, data.max.y, k).y;
    }
  }

  // Set the w edge value to their closest voxel
  for (int j = data.min.y; j <= data.max.y; j++) {
    for (int i = data.min.x; i <= data.max.x; i++) {
      mac->w(i, j, data.min.z) = src->value(i, j, data.min.z).z;
      mac->w(i, j, data.max.z + 1) = src->value(i, j, data.max.z).z;
    }
  }

  // Loop over internal u values
  for (int k = data.min.z; k <= data.max.z; ++k) {
    for (int j = data.min.y; j <= data.max.y; ++j) {
      for (int i = data.min.x + 1; i <= data.max.x; ++i) {
        mac->u(i, j, k) = 
          (src->value(i, j, k).x + src->value(i - 1, j, k).x) * 0.5;
      }
    }
  }

  // Loop over internal v values
  for (int k = data.min.z; k <= data.max.z; ++k) {
    for (int j = data.min.y + 1; j <= data.max.y; ++j) {
      for (int i = data.min.x; i <= data.max.x; ++i) {
        mac->v(i, j, k) = 
          (src->value(i, j, k).y + src->value(i, j - 1, k).y) * 0.5;
      }
    }
  }

  // Loop over internal w values
  for (int k = data.min.z + 1; k <= data.max.z; ++k) {
    for (int j = data.min.y; j <= data.max.y; ++j) {
      for (int i = data.min.x; i <= data.max.x; ++i) {
        mac->w(i, j, k) = 
          (src->value(i, j, k).z + src->value(i, j, k - 1).z) * 0.5;
      }
    }
  }
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
