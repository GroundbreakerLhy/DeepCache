PYTHON_BIN=/home/groundbreaker/miniconda3/envs/deepcache/bin/python
TVM_HOME=/home/groundbreaker/miniconda3/envs/deepcache/lib/python3.10/site-packages/tvm
CFLAGS = -O3 -no-pie -g
LDFLAGS = -L$(TVM_HOME) -ltvm_runtime -ldl -lpthread -lrt -Wl,-rpath,$(TVM_HOME)
INCLUDES = -I../../../include

TARGET = build/demo_static

.PHONY: all clean

all: $(TARGET)

$(TARGET): build/model.so demo_static.c
	gcc $(CFLAGS) demo_static.c -o $@ $(INCLUDES) $(LDFLAGS)

build/model.so: build_model.py
	$(PYTHON_BIN) build_model.py

clean:
	rm -rf build/