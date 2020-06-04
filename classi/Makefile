all: CurveLaneDetection

hello: main.o CurveLaneDetection.o
	g++ main.o CurveLaneDetection.o -o CurveLaneDetection `pkg-config --cflags --libs opencv4`

main.o: main.cpp
	g++ -c main.cpp `pkg-config --cflags --libs opencv4`

CurveLaneDetection.o: CurveLaneDetection.cpp
	g++ -c CurveLaneDetection.cpp `pkg-config --cflags --libs opencv4`
