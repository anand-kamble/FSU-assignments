CC = g++

CFLAGS = -Wall -std=c++23

SRC_DIR = src
BIN_DIR = bin

MAIN_SRC = $(SRC_DIR)/main.cpp
SPARSE_SRC = $(SRC_DIR)/Sparse.cpp

EXE_NAME = a.out
EXE_PATH = $(BIN_DIR)/$(EXE_NAME)

all: $(EXE_PATH) 

$(EXE_PATH): 
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(MAIN_SRC)
	@echo  "----------------------------------------------------------------"
	@echo  "You can execute the program by running : "  $@

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR)

.PHONY: all clean test