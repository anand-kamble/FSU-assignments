using namespace std;
template<typename T>
bool arrayIncludesNullPointer(T*arr, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        if (arr[i] == nullptr)
        {
            return true; // Found a nullptr in the array
        }
    }
    return false; // No nullptr found in the array
}
