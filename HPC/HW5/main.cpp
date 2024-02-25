/**
 * @file main.cpp
 * @brief This file contains the main function for a prime number gap calculation using OpenMP in binary tree.
 *
 * This program calculates the maximum prime gap within a specified range [n, m].
 * It utilizes OpenMP for parallelization to enhance performance for various thread counts.
 *
 * @name Author: Anand Kamble
 * @date Date: 20th Feb 2024
 *
 * @note Make sure to include the necessary dependencies:
 *   - iostream
 *   - cmath
 *   - omp.h
 *
 * @warning Ensure proper compilation with OpenMP support for parallel execution.
 */

#include <iostream>
#include <cmath>
#include <omp.h>
#include <vector>

using namespace std;

// Binary tree node structure
struct TreeNode
{
	long long int data;
	long long int level;
	long long int largestGapInNode;
	TreeNode *left;
	TreeNode *right;
};

// Function to create a new node
TreeNode *newNode(long long int value, long long int level)
{
	TreeNode *node = new TreeNode;

	node->data = value;
	node->level = level;
	node->left = nullptr;
	node->right = nullptr;
	node->largestGapInNode = 0;

	return node;
}

// Function to check if a number is prime
bool isPrime(long long int num)
{
	if (num <= 1)
		return false;
	if (num <= 3)
		return true;
	if (num % 2 == 0 || num % 3 == 0)
		return false;

	/**
	 * I did try to parallelize this part as well, but it was slowing down the whole code.
	 * It might be due to the creating of so many parallel tasks.
	 */
	for (long long int i = 5; i * i <= num; i += 6)
	{
		if (num % i == 0 || num % (i + 2) == 0)
			return false;
	}
	return true;
}

// Function to insert primes in the binary tree
/**
 * I have added a level parameter to keep track of the level of the node in the tree.
 */

long long int insertPrimes(TreeNode *&root, long long int n, long long int m, long long int level, long long int &prev)
{

	if (n > m)
		return 0;

	long long int mid = (n + m) / 2;
	long long int left_gap = 0;
	long long int right_gap = 0;

	if (isPrime(mid))
	{
		root = newNode(mid, level);
		auto a = insertPrimes(root->left, n, mid - 1, level + 1, prev);
		auto b = insertPrimes(root->right, mid + 1, m, level + 1, prev);
		return max(max(mid - n, m - mid), max(a, b));
	}
	else
	{
		long long int leftPrime = mid - 1;
		long long int rightPrime = mid + 1;

		/**
		 * Here I tried to using two threads as master if the number of threads is greater than 8.
		 * but then the speed up I was getting was in of the factor of 30 to 40 times.
		 * Which didn't seem right. That's why I commented it out.
		 *
		 * Here are the times I got with 2 master threads
		 * threads = [1, 2, 4, 8, 12]
		 * times = [0.022040, 0.014552, 0.008940, 0.000268, 0.000375]
		 */

		// If the number of threads is greater than or equal to 8, we can create two masters for the left and right subtrees.
		// 		if (omp_get_max_threads() >= 8)
		// 		{
		// #pragma omp single nowait
		// 			{
		// 				searchLeft(root, leftPrime, n, m, level, left_gap, L_a, L_b);
		// 			}
		// #pragma omp single nowait
		// 			{
		// 				searchRight(root, rightPrime, n, m, level, right_gap, R_a, R_b);
		// 			}
		// 		}

		long long int L_a = 0;
		long long int L_b = 0;
		// Search for prime numbers towards left
		while (leftPrime >= n)
		{
			if (isPrime(leftPrime))
			{

				root = newNode(leftPrime, level);
				// Since every node is creating two child tasks, we need to check if the number of threads is less than 2 * level.
				if (2 * root->level > omp_get_max_threads())
				// if (false)
				{
					left_gap = max(insertPrimes(root->left, n, leftPrime - 1, level + 1, prev),
								   insertPrimes(root->right, leftPrime + 1, m, level + 1, prev));
				}
				else
				{
#pragma omp task
					{
						L_a = insertPrimes(root->left, n, leftPrime - 1, level + 1, prev);
						L_b = insertPrimes(root->right, leftPrime + 1, m, level + 1, prev);
						left_gap = max(L_a, L_b);
					}
				}

				break;
			}
			leftPrime--;
		}

		long long int R_a = 0;
		long long int R_b = 0;
		while (rightPrime <= m)
		{
			if (isPrime(rightPrime))
			{

				root = newNode(rightPrime, level);
				// Since every node is creating two child tasks, we need to check if the number of threads is less than 2 * level.
				if (2 * root->level > omp_get_max_threads())
				// if (false)
				{
					right_gap = max(insertPrimes(root->left, n, rightPrime - 1, level + 1, prev),
									insertPrimes(root->right, rightPrime + 1, m, level + 1, prev));
				}
				else
				{
#pragma omp task
					{
						R_a = insertPrimes(root->left, n, rightPrime - 1, level + 1, prev);
						R_b = insertPrimes(root->right, rightPrime + 1, m, level + 1, prev);
						right_gap = max(R_a, R_b);
					}
				}

				break;
			}
			rightPrime++;
		}
#pragma omp taskwait
		return max((rightPrime - leftPrime), max(left_gap, right_gap));
	}
}
// pass the root node of your binary tree

// Function to prlong long int inorder traversal of the tree
void inorderTraversal(TreeNode *root)
{
	if (root == nullptr)
		return;
	inorderTraversal(root->left);
	cout << root->data << " ";
	inorderTraversal(root->right);
}

int main()
{
	long long int n, m;
	cout << "Enter the range [n, m]: ";
	cin >> n >> m;

	TreeNode *root = nullptr;

	int threads[] = {1, 2, 3, 4, 5, 6};
	// int threads[] = {12};
	int max_threads = omp_get_max_threads();
	cout << "Number of threads available: " << max_threads << endl;

	long long int gap = 0;

	for (auto NUM_THREADS : threads)
	{
		if (NUM_THREADS > max_threads)
			break;
		omp_set_num_threads(NUM_THREADS);

		double start = omp_get_wtime();

		long long int firstPrime = 2;
#pragma omp parallel
		{

// Letting one thread act as the master thread to create the tasks.
#pragma omp single nowait
			{
				gap = insertPrimes(root, n, m, 0, firstPrime);
			}
		}

		double end = omp_get_wtime();
		printf("Time for %2d threads : ", NUM_THREADS);
		cout << end - start << endl;
	}
	// inorderTraversal(root);

	cout << "Largest Prime gap for numbers between " << n << " and " << m << " is " << gap << endl;

	return 0;
}
