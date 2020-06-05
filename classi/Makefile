CC=g++

OC4CFLAGS = `pkg-config --cflags opencv4`
OC4LIBS = `pkg-config --libs opencv4`

CFLAGS = -I. -fPIC -frtti -pthread $(OC4CFLAGS)
LIBS = $(OC4LIBS)

OBJS := main.o \
	CurveLaneDetection.o

EXECUTABLE = CurveLaneDetection

all: $(OBJS) 
	$(CC) main.o CurveLaneDetection.o -o $(EXECUTABLE) $(LIBS)

main.o: main.cpp
	$(CC) -c main.cpp $(CFLAGS)

CurveLaneDetection.o: CurveLaneDetection.cpp
	$(CC) -c CurveLaneDetection.cpp $(CFLAGS)

clean:
	rm -f *.o 
	rm -f $(EXECUTABLE)

