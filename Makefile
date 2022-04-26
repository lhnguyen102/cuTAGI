# CUDA directory. This default is tested on Ubuntu 20.04
CUDA_ROOT_DIR=/usr/local/cuda

# CC compiler:
CC=g++ --std=c++14 
CC_FLAGS=
CC_LIBS=

# NVCC compiler:
NVCC=nvcc 
NVCC_FLAGS=
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart


# Source code:
SRC_DIR = src

# Obejct directory:
OBJ_DIR = bin

# Header file directory:
INC_DIR = include

# Application name:
EXE = main

# Object files:
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/indices.o $(OBJ_DIR)/net_prop.o\
$(OBJ_DIR)/feed_forward.o $(OBJ_DIR)/state_feed_backward.o\
$(OBJ_DIR)/param_feed_backward.o $(OBJ_DIR)/global_param_update.o\
$(OBJ_DIR)/common.o $(OBJ_DIR)/dataloader.o $(OBJ_DIR)/cost.o \
$(OBJ_DIR)/data_transfer.o $(OBJ_DIR)/task.o $(OBJ_DIR)/user_input.o\
$(OBJ_DIR)/net_init.o $(OBJ_DIR)/utils.o

## COMPILE

# Link c++ and CUDA compiled object files to application:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)



