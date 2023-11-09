#include <iostream>

template <typename T>
class BST {
private:
    struct Node {
        T data;
        Node* left;
        Node* right;

        Node(T value) : data(value), left(nullptr), right(nullptr) {}
    };

    Node* root;

    void insert(Node*& node, T value) {
        if (node == nullptr) {
            node = new Node(value);
        } else if (value < node->data) {
            insert(node->left, value);
        } else if (value > node->data) {
            insert(node->right, value);
        }
    }

	// Recursive function
    void inorder(Node* node) {
        if (node) {
            inorder(node->left);
            std::cout << node->data << " ";
            inorder(node->right);
        }
    }

public:
    BST() : root(nullptr) {}

    void insert(T value) {
        insert(root, value);
    }

    void display() {
        inorder(root);
        std::cout << std::endl;
    }

    // Destructor and other utility functions like delete, search, etc. can be added
};

int main() {
    BST<int> tree;
    tree.insert(50);
    tree.insert(30);
    tree.insert(70);
    tree.insert(20);
    tree.insert(40);
    tree.insert(60);
    tree.insert(80);

    tree.display(); // Should display: 20 30 40 50 60 70 80
    return 0;
}

