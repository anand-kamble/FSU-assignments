
#ifndef __BINARY_TREE_H__
#define __BINARY_TREE_H__

struct Node
{
	Node* left;
	Node* right;
	int data;

	Node() {
		left = NULL;
		right = NULL;
		data = 0;
	}
};

class BinaryTree
{
private:
	Node* root;
public:
	BinaryTree();
	~BinaryTree();
	BinaryTree(const BinaryTree&);
	const BinaryTree&  operator=(const BinaryTree&);

	void insert(int d);  // insert d into the tree
	void remove(int d);  // insert d into the tree
	//Node& find(int d);  // the return value must exist
	Node* find(int d);  // return NULL if integer d not found
	void print();

private:
	void insert(int d, Node* node);  // insert d into the tree
	Node* find(int f, Node* leaf);
	void printDepth(Node* node);
};

#endif
