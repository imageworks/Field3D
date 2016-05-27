//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3DExt_FrustumBuilder_H_
#define _INCLUDED_Field3DExt_FrustumBuilder_H_

//----------------------------------------------------------------------------//

#include <boost/shared_ptr.hpp>

#include "ns.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// FrustumBuilder
//----------------------------------------------------------------------------//

//! Base class for building frustums. In each subclass, one would implement
//! basic camera behavior that, given a bounding box, can determine multiple
//! screen-to-world and camera-to-world matrices that guarantee the bounding
//! of the provided box

class FrustumBuilder
{

public:

  // Typedefs ------------------------------------------------------------------

  typedef boost::shared_ptr<FrustumBuilder>       Ptr;
  typedef boost::shared_ptr<const FrustumBuilder> CPtr;

  typedef Field3D::FrustumFieldMapping FrustumMapping;

  // Ctors, dtor ---------------------------------------------------------------

  virtual ~FrustumBuilder()
  { }

  // To be implemented by subclasses -------------------------------------------

  //! Configures an existing frustum mapping instance such that it covers
  //! the provided bounding box over the course of the shutter.
  //! \param wsBounds The bounding box to cover.
  //! \param frustumMapping The mapping to configure.
  //! \param resolution The resolution to use with the resulting buffer.
  virtual bool setupFrustum(const Imath::Box3f &wsBounds, 
                            FrustumMapping::Ptr frustumMapping,
                            Imath::V3i &resolution) const = 0;

};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // include guard
