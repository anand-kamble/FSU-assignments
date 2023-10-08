# Compiler
CC = g++

# Compiler flags
CFLAGS = -Wall -std=c++23

# Directories
SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib
BIN_DIR = bin

# Source files  
MAIN_SRC = $(SRC_DIR)/main.cpp
TRIANGLE_SRC = $(SRC_DIR)/triangle_area.cpp
BRAIN_MESH_SRC = $(SRC_DIR)/brain_mesh.cpp

# Header files
TRIANGLE_HEADER = $(SRC_DIR)/triangle_area.h
BRAIN_MESH_HEADER = $(SRC_DIR)/brain_mesh.h

# Object files
MAIN_OBJ = $(OBJ_DIR)/main.o
TRIANGLE_OBJ = $(OBJ_DIR)/triangle_area.o
BRAIN_MESH_OBJ = $(OBJ_DIR)/brain_mesh.o

# Static libraries
TRIANGLE_NAME = libtriangle.a
BRAIN_MESH_NAME = libbrainmesh.a 
TRIANGLE_PATH = $(LIB_DIR)/$(TRIANGLE_NAME)
BRAIN_MESH_PATH = $(LIB_DIR)/$(BRAIN_MESH_NAME)

# Executable
EXE_NAME = a.out
EXE_PATH = $(BIN_DIR)/$(EXE_NAME)

# Targets
all: $(EXE_PATH) 

$(EXE_PATH): $(MAIN_OBJ) $(TRIANGLE_PATH) $(BRAIN_MESH_PATH)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(MAIN_OBJ) -L$(LIB_DIR) -lbrainmesh -ltriangle

$(MAIN_OBJ): $(MAIN_SRC)
	@mkdir -p $(OBJ_DIR) 
	$(CC) $(CFLAGS) -c -o $@ $(MAIN_SRC)

$(TRIANGLE_PATH): $(TRIANGLE_OBJ) 
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $(TRIANGLE_OBJ)

$(BRAIN_MESH_PATH): $(BRAIN_MESH_OBJ)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $(BRAIN_MESH_OBJ) $(TRIANGLE_OBJ)

$(TRIANGLE_OBJ): $(TRIANGLE_SRC) $(TRIANGLE_HEADER)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $(TRIANGLE_SRC)

$(BRAIN_MESH_OBJ): $(BRAIN_MESH_SRC) $(BRAIN_MESH_HEADER)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $(BRAIN_MESH_SRC)

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR)

.PHONY: all clean