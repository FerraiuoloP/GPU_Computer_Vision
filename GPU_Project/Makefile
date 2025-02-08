# Variables
CCBIN = g++
CUDA_STD = -std=c++14
PKG_CONFIG = $(shell pkg-config --cflags --libs opencv4)
NVCC = nvcc -arch=sm_75

# Source files and output
SRC_DIR = src
ifdef CPU 
MAIN = main_cpu.cpp
SOURCE_FILES = $(MAIN) $(SRC_DIR)/utils.cpp $(SRC_DIR)/edge_detection_cpu.cpp 
OUTPUT_FILE = build/main_cpu
else
MAIN = main.cpp
SOURCE_FILES = $(MAIN) $(SRC_DIR)/cuda_kernel.cu $(SRC_DIR)/utils.cpp  $(SRC_DIR)/cuda_otsu.cu
OUTPUT_FILE = build/main
endif

# Default target
all: $(OUTPUT_FILE)

# Build the executable
$(OUTPUT_FILE): $(SOURCE_FILES)
	mkdir -p build
	$(NVCC) $(CUDA_STD) -ccbin $(CCBIN) $(SOURCE_FILES) -o $(OUTPUT_FILE) $(PKG_CONFIG)

# Run the program
run: $(OUTPUT_FILE)
	./$(OUTPUT_FILE) $$ARGS
run-all: $(OUTPUT_FILE)
	./$(OUTPUT_FILE) -H $$ARGS > /dev/null &
	./$(OUTPUT_FILE) -C $$ARGS > /dev/null &
	./$(OUTPUT_FILE) -O $$ARGS > /dev/null &

# Clean the build directory
clean:
	rm -rf build

all_run: clean all run

all_run_cpu: clean all run_cpu




# Phony targets
.PHONY: all run clean