#include <iostream>
#include <stdlib.h>
#include <vector>
using namespace std;

// create the node structure for an Linked List
class Node
{
public:
    int data;
    Node *prev;
    Node *next;

    // constructor
    Node(int i) : data(i), prev(NULL), next(NULL) {}

	// destructor
	~Node() {}
};

class DoubleLinkedList
{
private:
    Node *head;
    Node *tail;

public:
    // Create root node

    // Constructor
    DoubleLinkedList() : head(NULL), tail(NULL) {}

    Node *getHead()
    {
        return head;
    }

    Node *getTail()
    {
        return tail;
    }

	// Add a new node at the end of the list
    void append(int data)
    {
        Node *new_node = new Node(data);
		if (!head)   // if head is NULL, then tail is NULL
		{
			head = tail = new_node;
		} else {
			tail->next = new_node;
			new_node->prev = tail;
			tail = new_node;
		}
	}

	// Add a new node before the head of the list
	void prepend(int data)
	{
		Node* new_node = new Node(data);
		// empty list
		if (!head) {
			head = tail = new_node;
		} else {
			new_node->next = head; 
			head->prev = new_node;
			head = new_node;
		}
	}

    // Delete the first node that contains data
    void deleteNode(int data)
    {
		// find node to delete
		Node* cur_node = head;
		while (cur_node && cur_node->data != data)
		{
			cur_node = cur_node->next;
		}
		// The cur_node is such that the next node contains data

     	if (!cur_node)   // cur_node == NULL
		{
			std::cout << "Node with data " << data << " not found." << std::endl;
 			return;
		}

		// When is cur_node->prev == NULL? Possibly never. 
		if (cur_node->prev) 
		{
			cur_node->prev->next = cur_node->next;
		} 
		else 
		{
			head = cur_node->next;
		}

		if (cur_node->next) 
		{
			cur_node->next->prev = cur_node->prev;
		} 
		else 
		{
			tail = cur_node->prev;
		}

		delete cur_node;
	}

	~DoubleLinkedList()
	{
		Node* cur_node = head;
		while (cur_node) 
		{
			// Save next node before deleting the current node
			Node* nextNode = cur_node->next;
			delete cur_node;
			cur_node = nextNode;
		}
	}

    
	// Print the list
	friend ostream &operator<<(ostream &oss, const DoubleLinkedList &list);
};
    
ostream &operator<<(ostream &oss, const DoubleLinkedList &list)
{
	oss << "DoubleLinkedList data, from head node to tail node: ";
	for (Node *cur_node = list.head; cur_node; cur_node = cur_node->next)
	{
		oss << cur_node->data << " ";
	}
	return oss;
}
    
int main()
{
	// Test the list
	DoubleLinkedList mylist;
	mylist.append(3);
    cout << "append 3, " << mylist << endl;
	mylist.append(1);
    cout << "append 1, " << mylist << endl;
	mylist.prepend(7);
    cout << "prepend 7, " << mylist << endl;
	mylist.prepend(9);
    cout << "prepend 9, " << mylist << endl;

	mylist.deleteNode(7);
    cout << "delete 7, " << mylist << endl;
	mylist.deleteNode(3);
    cout << "delete 3, " << mylist << endl;
	mylist.deleteNode(9);
    cout << "delete 9, " << mylist << endl;
	mylist.deleteNode(1);
    cout << "delete 1, " << mylist << endl;
	mylist.deleteNode(1);
}
