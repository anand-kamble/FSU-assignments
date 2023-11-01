# Compiler
CC = g++

# Compiler flags
CFLAGS = -Wall -std=c++23

# Directories
SRC_DIR = src
BIN_DIR = bin

# Source files  
MAIN_SRC = $(SRC_DIR)/main.cpp
NEWTON_SRC = $(SRC_DIR)/Classes/newton.cpp
SECANT_SRC = $(SRC_DIR)/Classes/secant.cpp

# Executable
EXE_NAME = a.out
EXE_PATH = $(BIN_DIR)/$(EXE_NAME)

# Targets
all: $(EXE_PATH) 

$(EXE_PATH): 
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(MAIN_SRC) -o $@ 
	@echo  "----------------------------------------------------------------"
	@echo  "You can execute the program by running : "  $@


clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR)

.PHONY: all clean test