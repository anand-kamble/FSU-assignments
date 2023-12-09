#include "stdio.h"
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <omp.h> 

void random_ints(std::vector<int>& random_numbers, long n=1000000) {
    const int max_value = 256;

    // Step 1: Create a weighted distribution
    std::vector<int> weighted_distribution;
    for (int i = 0; i <= max_value; i++) {
        // 2nd argument of insert() is the number of times 
        // the 3rd argument is inserted into the list
        weighted_distribution.insert(weighted_distribution.end(), i, i);
    }

    // Step 2: Randomly sample from the distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, weighted_distribution.size() - 1);

    for (long i=0; i < n; i++) {
        int num = weighted_distribution[dis(gen)];
        random_numbers.push_back(num);
    }
}
//----------------------------------------------------------------------
int main()
{
	int i;
    long m;
    long n = 100000000;
    std::vector<unsigned char> s(n);
    std::vector<int> random_numbers;

    // generate random integers between 0 and 255 
    random_ints(random_numbers, n);

    for (int i=0; i < 10; i++) {
        printf("random_ints[%d]: %d\n", i, random_numbers[i]);
    }

    std::vector<int> num(256);
    // initialize all elements to zero
    std::fill(num.begin(), num.end(), 0);

    // Loop to parallelize
	for (i = 0; i < n; i++)
	{
		num[random_numbers[i]]++;
	}

	for (i = 0; i<256; i++) {
		printf("the char %d freq is %f\n", i, num[i]/(double) 256);
    }
}
