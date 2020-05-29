all:
	g++ curve_lane_detection.cpp -o curve_lane_detection `pkg-config --cflags --libs opencv4`
compile:
