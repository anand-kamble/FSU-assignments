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

# Executable
EXE_NAME = test.x
EXE_PATH = $(BIN_DIR)/$(EXE_NAME)

# Targets
all: $(EXE_PATH) 

$(EXE_PATH):
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(MAIN_SRC) -o $(EXE_PATH)

run: $(EXE_PATH) 
	@echo "Running the program"
	@mpirun --use-hwthread-cpus  -np 12 $(EXE_PATH) 

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean run
