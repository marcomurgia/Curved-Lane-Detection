CC=g++

OC4CFLAGS = `pkg-config --cflags opencv4`
OC4LIBS = `pkg-config --libs opencv4`

CFLAGS = -I. -fPIC -frtti -pthread $(OC4CFLAGS)
LIBS = $(OC4LIBS)

OBJS := main.o \
        CurveLaneDetection.o

EXECUTABLE = CurveLaneDetection
OBJDIR = bin
OPTIMIZATION = -Os 

all: $(OBJS)
	$(CC) $(OBJDIR)/main.o $(OBJDIR)/CurveLaneDetection.o -o $(OBJDIR)/$(EXECUTABLE) $(LIBS)

main.o: main.cpp
	$(CC) -c $(OPTIMIZATION) main.cpp -o $(OBJDIR)/main.o $(CFLAGS)

CurveLaneDetection.o: CurveLaneDetection.cpp
	$(CC) -c $(OPTIMIZATION) CurveLaneDetection.cpp -o $(OBJDIR)/CurveLaneDetection.o $(CFLAGS)

clean:
	rm -f $(OBJDIR)/*.o
	rm -f $(OBJDIR)/$(EXECUTABLE)

