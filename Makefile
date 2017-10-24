.PHONY=all tests clean obj
CXX=g++
CC=gcc
WARNINGS=-Wall -Wextra -Wno-char-subscripts \
		 -Wpointer-arith -Wwrite-strings -Wdisabled-optimization \
		 -Wformat -Wcast-align -Wno-unused-function -Wno-unused-parameter
DBG:= -DNDEBUG
OPT:= -O3 -funroll-loops -pipe -fno-strict-aliasing -march=native -fopenmp -DUSE_PDQSORT
OS:=$(shell uname)
STD:=c++17
ifneq (,$(findstring g++,$(CXX)))
	ifeq ($(shell uname),Darwin)
		ifeq (,$(findstring clang,$(CXX)))
			STD :=c++17
			FLAGS := $(FLAGS) -Wa,-q
		else
			STD :=c++1z
			FLAGS := $(FLAGS) -flto
			CLHASH_CHECKOUT := "&& git checkout master"
		endif
	else
		FLAGS := $(FLAGS) -flto -std=c++1z
	endif
endif
OPT:=$(OPT) $(FLAGS)
FLOAT_TYPE=double
XXFLAGS=-fno-rtti
CXXFLAGS=$(OPT) $(XXFLAGS) -std=$(STD) $(WARNINGS) -DFLOAT_TYPE=$(FLOAT_TYPE)
CCFLAGS=$(OPT) -std=c11 $(WARNINGS)
LIB=-lz -lopenblas -lfftw3
LD=-L. -L/opt/local/lib

OBJS=$(patsubst %.cpp,%.o,$(wildcard lib/*.cpp))
TEST_OBJS=$(patsubst %.cpp,%.o,$(wildcard test/*.cpp))
EXEC_OBJS=$(patsubst %.cpp,%.o,$(wildcard src/*.cpp))

EX=$(patsubst src/%.o,%,$(EXEC_OBJS))

# If compiling with c++ < 17 and your compiler does not provide
# bessel functions with c++14, you must compile against boost.

INCLUDE=-I.

OBJS:=$(OBJS)

all: libwht.a $(OBJS) $(EX)

obj: $(OBJS) $(EXEC_OBJS)

spiral-wht-1.8: spiral-wht-1.8.tgz
	tar -zxf $<

libwht.a: spiral-wht-1.8
	cd spiral-wht-1.8 && ./configure && make && cp libwht.a .. && cd ..

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DBG) $(INCLUDE) $(LD) -c $< -o $@ $(LIB)

%: src/%.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(DBG) $(INCLUDE) $(LD) $(OBJS) $< -o $@ $(LIB)

#test: src/test.cpp $(OBJS)
#	$(CXX) $(CXXFLAGS) $(DBG) $(INCLUDE) $(LD) $(OBJS) $< -o $@ $(LIB)

%.o: %.c
	$(CC) $(CCFLAGS) -Wno-sign-compare $(DBG) $(INCLUDE) $(LD) -c $< -o $@ $(LIB)


tests: clean

clean:
	rm -f $(EXEC_OBJS) $(OBJS) $(EX) $(TEST_OBJS) lib/*o src/*o

mostlyclean: clean
