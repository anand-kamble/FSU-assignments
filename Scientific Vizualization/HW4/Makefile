compiler = g++
flags = -std=c++17

libs = -lglfw -lGLEW -lGL 

source = ./src/main.cpp ./src/glad.c ./src/stb_image.cpp
output = bin/main.x

all:
	$(compiler) $(flags) $(source) -o $(output) $(libs)

run: all
	./$(output)

clean:
	rm -rf $(output)