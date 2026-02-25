CC = gcc
CFLAGS = -O2 -Wall

LDFLAGS = -lrt

all: llc_attacker

llc_attacker: llc_attacker.c cacheutils.h functions.h multihist.h
	$(CC) $(CFLAGS) -o $@ llc_attacker.c $(LDFLAGS)

clean:
	rm -f llc_attacker
