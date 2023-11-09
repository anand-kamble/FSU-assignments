#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <assert.h>
#include "binary_tree.h"

using namespace std;

int main()
{
	BinaryTree bt;
	vector<int> rand_f;
	// time in seconds? I would like time in nanoeconds
	srand(std::time(nullptr)); // seed

	int nb_rand = 12;
	for (int i=0; i < nb_rand; i++) {
		int rnd = rand() / 1000000;
		printf("insert rnd= %d\n", rnd);
		rand_f.push_back(rnd);
		bt.insert(rnd);
	}

	// I expect to find this number
	assert(bt.find(rand_f[10])->data == rand_f[10]);

	// I do not expect to find 5
	assert(bt.find(5) == NULL);

	// print the tree
	bt.print();
   
	return 0;
}
