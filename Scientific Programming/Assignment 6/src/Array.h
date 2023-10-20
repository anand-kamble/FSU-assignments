class Array
{
public:
    int *data;

    Array();

    ~Array();

    Array(int size);

    Array(const Array &);

    Array &operator=(const Array &);

    void push_back(int value);

    void pop_back();

    void remove(int index);

    void insert(int value, int index);

    int getCapacity() const;
    int size() const;

    void clear();

private:
    int numberOfElements;

    int capacity;
};
