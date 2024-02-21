CC = g++

PETSC_DIR = /opt/petsc/3.20.4
PETSC_CFLAGS = -I$(PETSC_DIR)/include
PETSC_LDFLAGS = -L$(PETSC_DIR)/lib \
			-lm \
			-lpetsc
OPENCV_LIBS = `pkg-config --cflags --libs opencv4`


TARGET = main

INCLUDES = inc
SOURCE = src/OsmosisInpainting.cpp \
		 src/LinearOsmosis.cpp\


$(TARGET): $(SOURCE)
	$(CC) $(PETSC_CFLAGS) -o $(TARGET) $(SOURCE) $(PETSC_LDFLAGS) $(OPENCV_LIBS) -I$(INCLUDES)

clean:
	rm -f $(TARGET)

.PHONY: all run clean