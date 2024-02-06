# Compiler
CC = g++

# Compiler flags
CFLAGS = -Wall -std=c++23 -fopenmp

# Directories
SRC_DIR = src
BIN_DIR = bin

# Source files  
MAIN_SRC = $(SRC_DIR)/main.cpp

# Executable
EXE_NAME = test.x
EXE_PATH = $(BIN_DIR)/$(EXE_NAME)

# Targets
all: $(EXE_PATH) 

$(EXE_PATH):
	@mkdir -p $(BIN_DIR)
	# $(CC) $(CFLAGS) -o $@ $(MAIN_OBJ) -L$(LIB_DIR)
	$(CC) $(CFLAGS) $(MAIN_SRC) -o $(EXE_PATH)

run: $(EXE_PATH) 
	@echo "Running the program"
	@$(EXE_PATH)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean run
