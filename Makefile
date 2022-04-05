# --- MACROS
MAIN = conv
COMBINER = combiner
SPLITTER = splitter

# define the C compiler to use
MPICC = mpicc
NVCC = nvcc
CC = gcc

SRCDIR = src
OBJDIR = build

# add -ltiff here if tiff images are to be used
LIBS = -lm -ltiff
CU_LIBS = -lcudart -lcudadevrt

ifndef KERNEL_SCRIPT
	KERNEL_SCRIPT = processingScripts/processing.cu
endif

# path to config file
CONF_FLAGS = -Isrc/configs
# define any compile-time flags
CFLAGS = -Wall -pedantic -lpthread $(CONF_FLAGS) -fopenmp
NVCC_FLAGS = -lpthread -rdc=true $(CONF_FLAGS)

ifndef RELEASE
	CFLAGS += -pg -g
	NVCC_FLAGS += -pg -g -G
else
	CFLAGS += -O3
	NVCC_FLAGS += -O3 -Xptxas -O3
endif


# main files to be compiled with gcc

SPLITTER_MAIN_SRC = src/splitterMain.c
SPLITTER_MAIN_OBJ = $(SPLITTER_MAIN_SRC:%.c= $(OBJDIR)/%.o)

COMBINER_MAIN_SRC = src/combinerMain.c
COMBINER_MAIN_OBJ = $(COMBINER_MAIN_SRC:%.c= $(OBJDIR)/%.o)

CONV_MAIN_SRC = src/convMain.c
CONV_MAIN_OBJ = $(CONV_MAIN_SRC:%.c= $(OBJDIR)/%.o)

# c files to be compiled with gcc
C_SRCS_B = util/util.c imageFormats/pgmformat.c imageFormats/tiffformat.c imageFormats/kernelformat.c job/job.c
C_SRCS = $(addprefix src/, $(C_SRCS_B))
C_OBJS = $(C_SRCS:%.c= $(OBJDIR)/%.o)

# c files to be compiled with mpicc
MPI_SRCS_B = worker.c master.c jobExecution.c
MPI_SRCS = $(addprefix src/mpi/, $(MPI_SRCS_B))
MPI_OBJS = $(MPI_SRCS:%.c= $(OBJDIR)/%.o)

# cuda files to be compiled with nvcc
CU_SRCS_B = util/cudaUtils.cu cudaInvoker.cu $(KERNEL_SCRIPT) kernels/convKernels.cu io/fastpgm.cu io/fastBufferIO.cu cudaInvokerAsync.cu
CU_SRCS = $(addprefix src/cuda/, $(CU_SRCS_B))
CU_OBJS = $(CU_SRCS:%.cu=$(OBJDIR)/%.cu.o)
CU_LINK_OBJS = $(CU_SRCS:%.cu=$(OBJDIR)/%.cu.link.o)

# --- TARGETS

all: $(MAIN) $(COMBINER) $(SPLITTER)

$(SPLITTER): $(SPLITTER_MAIN_SRC) $(C_OBJS)
	@mkdir -p $(@D)
	@echo #
	@echo "-- CREATING SPLITTER --"
	$(CC) $(CFLAGS) $(MAIN_FLAGS) -o $@ $^ $(LIBS) -fopenmp

$(COMBINER): $(COMBINER_MAIN_SRC) $(C_OBJS)
	@echo #
	@echo "-- CREATING COMBINER --"
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS) -fopenmp

$(MAIN): $(CONV_MAIN_SRC) $(C_OBJS) $(MPI_OBJS) $(CU_OBJS) $(CU_LINK_OBJS)
	@echo #
	@echo "-- CREATING CONVOLUTION PROGRAM --"
	$(MPICC) $(CFLAGS) -o $@ $^ $(LIBS) $(CU_LIBS) -lstdc++


# c, mpi and cuda objects

$(C_OBJS): $(OBJDIR)/%.o : %.c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) -o $@ $<

$(MPI_OBJS): $(OBJDIR)/%.o : %.c
	@mkdir -p $(@D)
	$(MPICC) -c $(CFLAGS) -o $@ $<

$(CU_OBJS): $(OBJDIR)/%.cu.o : %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<
	$(NVCC) -dlink -o $(basename $@).link.o $@ -lcudart

clean:
	@echo #
	@echo "-- CLEANING PROJECT FILES --"
	$(RM) $(OBJDIR)/$(SRCDIR) -r $(MAIN) $(COMBINER) $(SPLITTER)