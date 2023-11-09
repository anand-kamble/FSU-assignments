// g++ without_stl.cpp

#include <iostream>

template <typename K, typename V>
class HashTable {
private:
    static const int TABLE_SIZE = 100;

    struct Entry {
        K key;
        V value;
        bool occupied;  // Flag to check if the entry is occupied
        Entry() : occupied(false) {}
    };

    Entry table[TABLE_SIZE];

    int hashFunction(const K& key) const {
        return key % TABLE_SIZE;
    }

public:
    HashTable() {}

    void insert(const K& key, const V& value) {
        int index = hashFunction(key);
        while (table[index].occupied && table[index].key != key) {
            index = (index + 1) % TABLE_SIZE;
        }
        table[index].key = key;
        table[index].value = value;
        table[index].occupied = true;
    }

    bool get(const K& key, V& value) {
        int index = hashFunction(key);
        int originalIndex = index;
        while (table[index].occupied) {
            if (table[index].key == key) {
                value = table[index].value;
                return true;
            }
            index = (index + 1) % TABLE_SIZE;
            if (index == originalIndex) break;  // We've checked the entire table
        }
        return false;
    }

    bool remove(const K& key) {
        int index = hashFunction(key);
        int originalIndex = index;
        while (table[index].occupied) {
            if (table[index].key == key) {
                table[index].occupied = false;
                return true;
            }
            index = (index + 1) % TABLE_SIZE;
            if (index == originalIndex) break;  // We've checked the entire table
        }
        return false;
    }
};

int main() {
    HashTable<int, std::string> hashTable;
    hashTable.insert(1, "Alice");
    hashTable.insert(2, "Bob");
    hashTable.insert(101, "Charlie");  // Causes a collision with key 1

    std::string name;
    if (hashTable.get(1, name)) {
        std::cout << "Key 1: " << name << std::endl;  // Outputs: Key 1: Alice
    }

    if (hashTable.get(101, name)) {
        std::cout << "Key 101: " << name << std::endl;  // Outputs: Key 101: Charlie
    }

    hashTable.remove(1);
    if (!hashTable.get(1, name)) {
        std::cout << "Key 1 not found." << std::endl;  // Outputs: Key 1 not found.
    }

    return 0;
}
