#include <iostream>
#include <cmath>

using namespace std;

// Binary tree node structure
struct TreeNode
{
	long long int data;
	TreeNode *left;
	TreeNode *right;
};

// Function to create a new node
TreeNode *newNode(long long int value)
{
	TreeNode *node = new TreeNode;
	node->data = value;
	node->left = nullptr;
	node->right = nullptr;
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

// Function to insert primes in the binary tree
// Function to insert primes in the binary tree
void insertPrimes(TreeNode *&root, long long int n, long long int m)
{
	if (n > m)
		return;

	long long int mid = (n + m) / 2;
	if (isPrime(mid))
	{
		root = newNode(mid);
		insertPrimes(root->left, n, mid - 1);
		insertPrimes(root->right, mid + 1, m);
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
				root = newNode(leftPrime);
				insertPrimes(root->left, n, leftPrime - 1);
				insertPrimes(root->right, leftPrime + 1, m);
				return;
			}
			leftPrime--;
		}

		// Search for prime numbers towards right
		while (rightPrime <= m)
		{
			if (isPrime(rightPrime))
			{
				root = newNode(rightPrime);
				insertPrimes(root->left, n, rightPrime - 1);
				insertPrimes(root->right, rightPrime + 1, m);
				return;
			}
			rightPrime++;
		}
	}
}

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

	insertPrimes(root, n, m);

	cout << "Prime numbers between " << n << " and " << m << " are: ";
	inorderTraversal(root);
	cout << endl;

	return 0;
}
