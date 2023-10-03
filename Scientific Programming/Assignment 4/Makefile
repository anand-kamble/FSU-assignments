# Compiler
CC = g++

# Compiler flags
CFLAGS = -Wall -std=c++11

# Directories
SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib
BIN_DIR = bin


# Source files
MAIN_SRC = $(SRC_DIR)/main.cpp
TRIANGLE_SRC = $(SRC_DIR)/triangle_area.cpp

# Object files
MAIN_OBJ = $(OBJ_DIR)/main.o
TRIANGLE_OBJ = $(OBJ_DIR)/triangle_area.o

# Static library
LIB_NAME = libtriangle.a
LIB_PATH = $(LIB_DIR)/$(LIB_NAME)

# Executable
EXE_NAME = main.out
EXE_PATH = $(BIN_DIR)/$(EXE_NAME)


# Targets
all: $(EXE_PATH)

$(EXE_PATH): $(MAIN_OBJ) $(LIB_PATH)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(MAIN_OBJ) -L$(LIB_DIR) -ltriangle

$(MAIN_OBJ): $(MAIN_SRC)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $(MAIN_SRC)

$(LIB_PATH): $(TRIANGLE_OBJ)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $(TRIANGLE_OBJ)

$(TRIANGLE_OBJ): $(TRIANGLE_SRC)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $(TRIANGLE_SRC)

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR)

.PHONY: all clean
