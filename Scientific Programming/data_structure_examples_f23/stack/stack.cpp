#include <iostream>

template <typename T>
class Stack {
private:
    struct Node {
        T data;
        Node* next;
    };

    Node* topNode = nullptr;  // only in C++11
    int count = 0;   // only in C++11

public:
    ~Stack() {
        while (!isEmpty()) {
            pop();
        }
    }

    // Push an element onto the stack
    void push(const T& value) {
        Node* newNode = new Node;
        newNode->data = value;
        newNode->next = topNode;
        topNode = newNode;
        count++;
    }

    // Pop an element from the stack
    T pop() {
        if (isEmpty()) {
            std::cerr << "Stack is empty! Cannot pop." << std::endl;
            exit(-1); // or throw an exception
        }

        Node* temp = topNode;
        T data = temp->data;
        topNode = topNode->next;
        delete temp;
        count--;

        return data;
    }

    // Return the top element without popping
    T top() const {
        if (isEmpty()) {
            std::cerr << "Stack is empty!" << std::endl;
            exit(-1); // or throw an exception
        }
        return topNode->data;
    }

    // Check if the stack is empty
    bool isEmpty() const {
        return topNode == nullptr;
    }

    // Return the number of elements in the stack
    int size() const {
        return count;
    }
};

int main() {
    Stack<int> s;

	int ints[] = {50, 10, 20, 40, 30};  

	std::cout << std::endl;
	for (auto i : ints) {
		std::cout << "push " << i << " onto the stack" << std::endl;
		s.push(i);
	}
	std::cout << std::endl;

    std::cout << "Top of the stack: " << s.top() << std::endl;
    std::cout << "Stack size: " << s.size() << std::endl;

	std::cout << std::endl;
    std::cout << "Popping elements:" << std::endl;
    while (!s.isEmpty()) {
        std::cout << s.pop() << std::endl;
    }

    return 0;
}
