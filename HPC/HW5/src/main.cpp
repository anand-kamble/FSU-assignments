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
	for (long long int i = 5; i * i <= num; i += 6)
	{
		if (num % i == 0 || num % (i + 2) == 0)
			return false;
	}
	return true;
}

// Declare a custom reduction for finding the maximum prime gap
#pragma omp declare reduction(primeMax : struct TreeNode * : omp_out = (omp_in->largestGapInNode > omp_out->largestGapInNode) ? omp_in : omp_out)

// Function to insert primes in the binary tree
// Function to insert primes in the binary tree
void insertPrimes(TreeNode *&root, long long int n, long long int m, long long int level, long long int largestGapInNode)
{
	if (n > m)
		return;

	long long int mid = (n + m) / 2;
	if (isPrime(mid))
	{
		root = newNode(mid, level);
		insertPrimes(root->left, n, mid - 1, level + 1, largestGapInNode);
		insertPrimes(root->right, mid + 1, m, level + 1, largestGapInNode);
		largestGapInNode = max(max(root->data - n, m - root->data), max(root->data - n, m - root->data));
	}
	else
	{
		long long int leftPrime = mid - 1;
		long long int rightPrime = mid + 1;

		// Search for prime numbers towards left
		while (leftPrime >= n)
		{
			if (isPrime(leftPrime))
			{

				root = newNode(leftPrime, level);
				if (root->level > 10)
				{
					insertPrimes(root->left, n, leftPrime - 1, level + 1, largestGapInNode);
					insertPrimes(root->right, leftPrime + 1, m, level + 1, largestGapInNode);
					largestGapInNode = max(max(root->data - n, m - root->data), max(root->data - n, m - root->data));
				}
				else
				{
#pragma omp task
					{
						insertPrimes(root->left, n, leftPrime - 1, level + 1, largestGapInNode);
					}
#pragma omp task
					{
						insertPrimes(root->right, leftPrime + 1, m, level + 1, largestGapInNode);
					}
					largestGapInNode = max(max(root->data - n, m - root->data), max(root->data - n, m - root->data));
					break;
				}
			}
			leftPrime--;
		}

		// Search for prime numbers towards right
		while (rightPrime <= m)
		{
			if (isPrime(rightPrime))
			{
				root = newNode(rightPrime, level);
				insertPrimes(root->left, n, rightPrime - 1, level + 1, largestGapInNode);
				insertPrimes(root->right, rightPrime + 1, m, level + 1, largestGapInNode);
				
				break;
			}
			rightPrime++;
		}
	}
}

void printBT(const std::string &prefix, const TreeNode *node, bool isLeft)
{
	if (node != nullptr)
	{
		std::cout << prefix;

		std::cout << (isLeft ? "├──" : "└──");

		// print the value of the node
		std::cout << node->data << std::endl;

		// enter the next tree level - left and right branch
		printBT(prefix + (isLeft ? "│   " : "    "), node->left, true);
		printBT(prefix + (isLeft ? "│   " : "    "), node->right, false);
	}
}

void printBT(const TreeNode *node)
{
	std::cout << "Binary Tree: " << std::endl;
	printBT("", node, false);
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
	// cout << "Enter the range [n, m]: ";
	// cin >> n >> m;

	n = 1;
	m = 10000;

	TreeNode *root = nullptr;

	int threads[] = {1, 2, 4, 8, 12, 16, 28, 32, 64};
	// int threads[] = {12};
	int max_threads = omp_get_max_threads();
	cout << "Number of threads available: " << max_threads << endl;

	for (auto NUM_THREADS : threads)
	{
		if (NUM_THREADS > max_threads)
			break;
		omp_set_num_threads(NUM_THREADS);

		double start = omp_get_wtime();

		long long int largestGapInNode = 0;

#pragma omp parallel reduction(max : largestGapInNode)
		{
#pragma omp single nowait
			{
				insertPrimes(root, n, m, 0, largestGapInNode);
			}
		}

		// inorderTraversal(root);
		// printBT(root);

		double end = omp_get_wtime();

		cout << "Time taken by " << NUM_THREADS << " threads : " << end - start << " seconds" << endl;
	}
	cout << "Prime numbers between " << n << " and " << m << endl;

	

	return 0;
}
