# edit this if your maya installation is somewhere else.
MAYA_LOCATION = /net/apps/spinux1/aw/maya2012

#edit this for the Field3D installation directory
FIELD3D_ROOT = /usr/3rd_party_software/Field3D
FIELD3D_INC = $(FIELD3D_ROOT)/include
FIELD3D_LIB = $(FIELD3D_ROOT)/lib

C++	 = g++

CFLAGS = -DBits64_ -m64 -DUNIX -D_BOOL -DLINUX -DFUNCPROTO -D_GNU_SOURCE \
         -DLINUX_64 -fPIC \
         -fno-strict-aliasing -DREQUIRE_IOSTREAM -Wno-deprecated -O3 -Wall \
         -Wno-multichar -Wno-comment -Wno-sign-compare -funsigned-char \
         -Wno-reorder -fno-gnu-keywords -ftemplate-depth-25 -pthread

C++FLAGS	= $(CFLAGS) $(WARNFLAGS) -Wno-deprecated -fno-gnu-keywords

INCLUDES	= -I. -I.. -I$(MAYA_LOCATION)/include \
						-I/usr/X11R6/include \
						-I/usr/include/OpenEXR \
						-I$(FIELD3D_INC)

LD	 = $(C++) -shared $(NO_TRANS_LINK) $(C++FLAGS)

# add more libs if you need to for your plugin
LIBS	 = -L$(MAYA_LOCATION)/lib -lOpenMaya\
				 -Wl,-rpath,$(FIELD3D_LIB) -L$(FIELD3D_LIB) -lField3D


exportF3d.so:	exportF3d.o
	-rm -f $@
	$(LD) -o $@ $^ $(LIBS) -lOpenMayaAnim -lOpenMayaFX


depend:
	makedepend $(INCLUDES) -I/usr/include/CC *.cc

clean:
	-rm -f *.o

Clean:
	-rm -f *.o *.so *.lib *.bak *.bundle

##################
# Generic Rules  #
##################

%.o :	%.cpp
	$(C++) -c $(INCLUDES) $(C++FLAGS) $<

%.$(so) :	%.cpp
	-rm -f $@
	$(C++) -o $@ $(INCLUDES) $(C++FLAGS) $< $(LFLAGS) $(LIBS)

%.$(so) :	%.o
	-rm -f $@
	$(LD) -o $@ $< $(LIBS)

