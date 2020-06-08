CC=g++

OC4CFLAGS = `pkg-config --cflags opencv4`
OC4LIBS = `pkg-config --libs opencv4`

CFLAGS = -I. -fPIC -frtti -pthread $(OC4CFLAGS)
LIBS = $(OC4LIBS)

OBJS := main.o \
	CurveLaneDetection.o

EXECUTABLE = CurveLaneDetection

OBJDIR = bin

all: $(OBJS) 
	$(CC) $(OBJDIR)/main.o $(OBJDIR)/CurveLaneDetection.o -o $(EXECUTABLE) $(LIBS)

main.o: main.cpp
	$(CC) -c main.cpp $(CFLAGS)
	mv main.o $(OBJDIR)

CurveLaneDetection.o: CurveLaneDetection.cpp
	$(CC) -c CurveLaneDetection.cpp $(CFLAGS)
	mv CurveLaneDetection.o $(OBJDIR)
clean:
	rm -f *.o 
	rm -f $(EXECUTABLE)

