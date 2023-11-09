#include <stdio.h>
#include <stdlib.h>
#include "binary_tree.h"

BinaryTree::BinaryTree() : root(NULL) {
}

BinaryTree::~BinaryTree() {
}

//BinaryTree::BinaryTree(const BinaryTree&) {}
//const BinaryTree&  BinaryTree::operator=(const BinaryTree&) {}

void BinaryTree::insert(int d) {
	// define root
	if (root == NULL) {
		root = new Node();
		root->data = d;
	} else {
		insert(d, root);
	}
} 

void BinaryTree::insert(int f, Node* leaf)
{
	//printf("insert\n");
	if (f <= leaf->data) {
		if (leaf->left != NULL) {
			insert(f, leaf->left);
		} else {
			leaf->left = new Node();
			leaf->left->data = f;
		}
	} else {
		if (leaf->right != NULL) {
			insert(f, leaf->right);
		} else {
			leaf->right = new Node();
			leaf->right->data = f;
		}
	}
}

void BinaryTree::remove(int d) {
}  

Node* BinaryTree::find(int d) { // ERROR? 
	if (root == NULL) {
	    printf("Integer <%d> not found\n", d);
		return NULL;
	}

	if (root->data == d) {
		return root;
	} else {
		return find(d, root);
	}
}

// Find first node where data == d (as an example), or return NULL
// leaf is always non-null
Node* BinaryTree::find(int d, Node* leaf)
{
	if (leaf->data == d) {
		return leaf;
	} else {
		if (leaf->left && d <= leaf->data) {
			return find(d, leaf->left);
		} else if (leaf->right && d > leaf->data) {
			return find(d, leaf->right);
		} else {
			printf("Int not found\n");
			return NULL;
		}
	}
}

void BinaryTree::print() {
 	printf("print()\n");
	printDepth(root);
}

void BinaryTree::printDepth(Node* node)
// print tree data starting at Node node
{
	if (node == NULL) {
		return;
	} else {
		printf("data: %d\n", node->data);
		printDepth(node->left);
		printDepth(node->right);
	}
}
//--------------------------------------------------------------------
