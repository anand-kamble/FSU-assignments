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
MAIN_SRC = $(SRC_DIR)/main_test.cpp
ARRAY_SRC = $(SRC_DIR)/Array.cpp

# Header files
ARRAY_HEADER = $(SRC_DIR)/Array.h

# Object files
MAIN_OBJ = $(OBJ_DIR)/main.o
ARRAY_OBJ = $(OBJ_DIR)/Array.o

# Static libraries
ARRAY_NAME = libArray.a 
ARRAY_PATH = $(LIB_DIR)/$(ARRAY_NAME)

# Executable
EXE_NAME = test.x
EXE_PATH = $(BIN_DIR)/$(EXE_NAME)

# Targets
all: $(EXE_PATH) 

test: MAIN_SRC = $(SRC_DIR)/test.cpp
test: $(EXE_PATH)

$(EXE_PATH): $(MAIN_OBJ) $(ARRAY_PATH)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(MAIN_OBJ) -L$(LIB_DIR) -lArray
	@echo  "----------------------------------------------------------------"
	@echo  "You can execute the program by running : "  $@

$(MAIN_OBJ): $(MAIN_SRC)
	@mkdir -p $(OBJ_DIR) 
	$(CC) $(CFLAGS) -c -o $@ $(MAIN_SRC)

$(ARRAY_PATH): $(ARRAY_OBJ)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $(ARRAY_OBJ) $(TRIANGLE_OBJ)

$(ARRAY_OBJ): $(ARRAY_SRC) $(ARRAY_HEADER)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $(ARRAY_SRC)

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR)

.PHONY: all clean test