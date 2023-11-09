// To compile: 
//     g++ -std=c++11 with_stl.cpp
//
#include <iostream>
#include <list>
#include <vector>
#include <string>

template <typename K, typename V>
class HashTable {
private:
    // Size of the hash table
    int capacity;
    // Number of key-value pairs in the hash table
    int size;

    // The table will be an array of lists to handle collisions using chaining
    std::vector<std::list<std::pair<K, V> > > table;

    int hashFunction(const K& key) const {
		// where should the hash be stored? 
        return std::hash<K>{}(key) % capacity;  // C++ 11 construct
    }

public:
    HashTable(int capacity = 100) : capacity(capacity), size(0) {
        table.resize(capacity);
    }

    void insert(const K& key, const V& value) {
        int index = hashFunction(key);
        for (auto& pair : table[index]) {     // C++11
            if (pair.first == key) {
                pair.second = value; // Update the value if key already exists
                return;
            }
        }
        table[index].push_back({key, value});      
        size++;
    }

    bool get(const K& key, V& value) {
        int index = hashFunction(key);
        for (auto& pair : table[index]) {    // C++11
            if (pair.first == key) {
                value = pair.second;
                return true;
            }
        }
        return false;
    }

    bool remove(const K& key) {
        int index = hashFunction(key);
        for (auto it = table[index].begin(); it != table[index].end(); ++it) {   // C++11
            if (it->first == key) {
                table[index].erase(it);
                size--;
                return true;
            }
        }
        return false;
    }

    int getSize() const {
        return size;
    }

    bool isEmpty() const {
        return size == 0;
    }
};

int main() {
    HashTable<std::string, int> hashTable;
    hashTable.insert("Alice", 25);
    hashTable.insert("Bob", 30);

    int age;
    if (hashTable.get("Alice", age)) {
		// Outputs: Alice\'s age: 25
        std::cout << "Alice's age: " << age << std::endl; 
    }

    hashTable.remove("Alice");
    if (!hashTable.get("Alice", age)) {
        std::cout << "Alice not found." << std::endl; // Outputs: Alice not found.
    }

    return 0;
}
