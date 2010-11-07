# Copyright (c) 2009 Sony Pictures Imageworks Inc. et al.
# 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the
# distribution. Neither the name of Sony Pictures Imageworks nor the
# names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written
# permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

# Author : Nicholas Yue yue.nicholas@gmail.com

# This module will define the following variables:
#  ILMBASE_INCLUDE_DIRS - Location of the ilmbase includes
#  ILMBASE_LIBRARIES - [TODO] Required libraries for all requested bindings
#  ILMBASE_FOUND - true if ILMBASE was found on the system
#  ILMBASE_LIBRARY_DIRS - the full set of library directories

FIND_PATH ( Ilmbase_Base_Dir include/OpenEXR/IlmBaseConfig.h
  ENV ILMBASE_ROOT
  )

IF ( Ilmbase_Base_Dir )

  SET ( ILMBASE_INCLUDE_DIRS
    ${Ilmbase_Base_Dir}/include
    ${Ilmbase_Base_Dir}/include/OpenEXR
    CACHE STRING "ILMBase include directories")
  SET ( ILMBASE_LIBRARY_DIRS ${Ilmbase_Base_Dir}/lib
    CACHE STRING "ILMBase library directories")
  SET ( ILMBASE_FOUND TRUE )

ENDIF ( Ilmbase_Base_Dir )
