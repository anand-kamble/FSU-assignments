#include <algorithm>

using namespace std;

typedef Function int (*)(int);

void error( (*cb)())
{
    cb();
}

int main()
{

    error([]()
          { return 1; });
    return 0;
}