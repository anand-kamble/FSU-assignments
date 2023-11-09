#include <iostream>
using namespace std;

int recursive_factorial(int n) 
{
	if (n < 0) {
		printf("n must be greater than zero\n");
	}

	if (n == 1 || n == 0) {
		return 1;
	}

	return n * recursive_factorial(n-1);
}


int nonrecursive_factorial(int n) 
//  Compute the factorial of n (arg) non-recursively
{
	int fact = 1; // 0!, or 1!
	for (int  i=2; i <= n; i++) {
		fact *= i;
	}
	return fact;
}

class Factorial 
{
public:
	int operator()(int n) {
		return nonrecursive_factorial(n);
	}
};

	

int main()
{
	Factorial f;
	// 120 = 5 * 4 * 3 * 2 * 1
	int fact;
	fact = recursive_factorial(5); 
	cout << "recursive fact(5) = " << fact << endl;

	fact = nonrecursive_factorial(5); 
	printf("fact= %d\n", fact);
	printf("fact= %d\n", f(1000));

	printf("fact(0)= %d\n", nonrecursive_factorial(0));
	printf("fact(1)= %d\n", nonrecursive_factorial(1));
}
