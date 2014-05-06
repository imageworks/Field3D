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

/*! \file PatternMatch.h
  \brief Contains functions for pattern matching field name/attributes.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_PatternMatch_H_
#define _INCLUDED_PatternMatch_H_

#include <vector>

// Boost includes
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <boost/thread/mutex.hpp>

#include "Field.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//

enum MatchFlags {
  MatchNoFlags = 0,
  MatchEmptyPattern = 1 << 0
};

//----------------------------------------------------------------------------//

//! Splits a string into a vector of strings, using ',' as the separator
std::vector<std::string>
split(const std::string &s);

//----------------------------------------------------------------------------//

//! Splits a string into a vector of strings, given separator characters
std::vector<std::string>
split(const std::string &s, const std::string &separatorChars);

//----------------------------------------------------------------------------//

//! Matches a <name>:<attribute> string against a set of patterns
bool 
match(const std::string &name, const std::string &attribute, 
      const std::vector<std::string> &patterns, 
      const MatchFlags flags = MatchEmptyPattern);
bool 
match(const std::string &name, const std::string &attribute, 
      const std::string &patterns, 
      const MatchFlags flags = MatchEmptyPattern);

//----------------------------------------------------------------------------//

//! Matches an <attribute> string against a set of patterns
bool 
match(const std::string &attribute, const std::vector<std::string> &patterns, 
      const MatchFlags flags = MatchEmptyPattern);
bool 
match(const std::string &attribute, const std::string &patterns, 
      const MatchFlags flags = MatchEmptyPattern);

//----------------------------------------------------------------------------//

//! Matches a field's name and attribute against a set of patterns.
bool 
match(const FieldRes *f, const std::vector<std::string> &patterns, 
      const MatchFlags flags = MatchEmptyPattern);
bool 
match(const FieldRes *f, const std::string &patterns, 
      const MatchFlags flags = MatchEmptyPattern);

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
