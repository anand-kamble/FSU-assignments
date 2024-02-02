compiler = g++
flags = -std=c++11

libs = -lglfw -lGLEW -lGL 

source = main.cpp glad.c
output = main.x

all:
	$(compiler) $(flags) $(source) -o $(output) $(libs)

run:
	./$(output)

clean:
	rm $(output)