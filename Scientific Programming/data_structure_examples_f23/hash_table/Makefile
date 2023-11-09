all: with_stl without_stl

with_stl:
	g++ -std=c++11 with_stl.cpp -o with_stl.x

without_stl:
	g++ without_stl.cpp -o without_stl.x

clean:
	rm -rf *.x *.o a.out

