#include <iostream>
#include <stdlib.h>
#include <vector>
using namespace std;

// create the node structure for an SingleLinked List
class Node
{
public:
    int data;
    Node *next;

    // constructor
    Node(int i) : data(i) {}

	// destructor
	~Node() {}
};

class SingleLinkedList
{
public:
    // Create root node
    Node *head;

    // Constructor
    SingleLinkedList() : head(NULL) {}

    Node *getHead()
    {
        return head;
    }

    void addNode(int data)
    {
        Node *new_node = new Node(data);
		if (head == 0) 
		{
			head = new_node;
		}

        // Get to the end node
		Node* cur_node = head;
        for ( ; cur_node && cur_node->next; cur_node = cur_node->next)
        {
            ;
        }
		cur_node->next = new_node;
		new_node->next = NULL;   // IMPORTANT
    }

    // Delete the first node that contains data
    void deleteNode(int data)
    {
        if (!head) {
            std::cout << "List is empty." << std::endl;
            return;
        }
    
        // Handle deletion of head separately
        if (head->data == data) {
            Node* temp = head;
            head = head->next;
            delete temp;
            return;
        }
    
        Node* cur_node = head;  // not empty
		// find cur_node such that cur_node->next == data
        for (; cur_node->next && cur_node->next->data != data; cur_node = cur_node->next)
            ;
    
        if (!cur_node->next) { // last node
            std::cout << "Node with data " << data << " not found" << std::endl;
            return;
        }
    
		// Without the next line, there'd be a memory leak
        Node* nodeToDelete = cur_node->next;
        cur_node->next = cur_node->next->next;
        delete nodeToDelete;
    }
    
	// Print the list
	friend ostream &operator<<(ostream &oss, const SingleLinkedList &list);
};
    
ostream &operator<<(ostream &oss, const SingleLinkedList &list)
{
	oss << "SingleLinkedList data, from head node to tail node: ";
	for (Node *cur_node = list.head; cur_node; cur_node = cur_node->next)
	{
		oss << cur_node->data << " ";
	}
	return oss;
}
    
int main()
{
	// Test the list
	SingleLinkedList mylist;
	mylist.addNode(3);
    cout << mylist << endl;
	mylist.addNode(1);
    cout << mylist << endl;
	mylist.addNode(7);
    cout << mylist << endl;
	mylist.addNode(9);
    cout << mylist << endl;

	mylist.deleteNode(7);
    cout << "delete 7, " << mylist << endl;
	mylist.deleteNode(3);
    cout << "delete 3, " << mylist << endl;
	mylist.deleteNode(33); // 33 not in the list
	mylist.deleteNode(9);
    cout << "delete 9, " << mylist << endl;
	mylist.deleteNode(1);
    cout << "delete 1, " << mylist << endl;
	mylist.deleteNode(1);
}
