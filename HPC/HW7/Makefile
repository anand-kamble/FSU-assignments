# Compiler
CC = mpiCC

# Compiler flags
CFLAGS = -std=c++17 

# Directories
SRC_DIR = src
BIN_DIR = bin
INC_DIR = includes

# Source files  
MAIN_SRC = $(SRC_DIR)/main.cpp

# Include files
INC_FILES = $(INC_DIR)/Jpegfile.cpp

# Executable
EXE_NAME = test.x
EXE_PATH = $(BIN_DIR)/$(EXE_NAME)

# Targets
all: $(EXE_PATH) 

$(EXE_PATH):
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(MAIN_SRC) $(INC_FILES) -o $(EXE_PATH) $(INC_DIR)/JpegLib/libjpeg.a

run: $(EXE_PATH) 
	@echo "Running the program"
	@mpirun -np 6 $(EXE_PATH)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean run
