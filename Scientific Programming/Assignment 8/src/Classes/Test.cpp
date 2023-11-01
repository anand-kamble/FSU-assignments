#include <iostream>
#include <math.h>
#include <string>

using namespace std;

class Test
{
private:
    string name;
    bool passed = false;
    float outputFloat = 0;
    int outputInt = 0;
    string outputString = "";
    const char *outputChar = NULL;
    int *matrix = new int[4];

    bool result()
    {
        cout << (this->passed ? "[Passed]" : "[Failed]") << " : " << this->name << endl;
        return this->passed;
    }

public:
    Test(string name)
    {
        this->name = name;
    };

    Test &Expect(float value)
    {
        this->outputFloat = value;
        this->passed = false;
        return *this;
    }

    Test &ToBe(float valueToBeExpected)
    {
        this->passed = fabs(this->outputFloat - valueToBeExpected) < 1.e-7;
        this->result();
        if (!this->passed)
            cout << "\tExpected: " << valueToBeExpected << "\n\tActual: " << this->outputFloat << endl;
        return *this;
    }

    Test &Expect(double value)
    {
        this->outputFloat = value;
        this->passed = false;
        return *this;
    }

    Test &ToBe(double valueToBeExpected)
    {
        this->passed = fabs(this->outputFloat - valueToBeExpected) < 1.e-14;
        this->result();
        if (!this->passed)
            cout << "\tExpected: " << valueToBeExpected << "\n\tActual: " << this->outputFloat << endl;
        return *this;
    }

    Test &Expect(int value)
    {
        this->outputInt = value;
        this->passed = false;
        return *this;
    }

    Test &ToBe(int valueToBeExpected)
    {
        this->passed = this->outputInt == valueToBeExpected;
        this->result();
        if (!this->passed)
            cout << "\tExpected: " << valueToBeExpected << "\n\tActual: " << this->outputInt << endl;
        return *this;
    }

    Test &Expect(string value)
    {
        this->outputString = value;
        this->passed = false;
        return *this;
    }

    Test &ToBe(string valueToBeExpected)
    {
        this->passed = this->outputString == valueToBeExpected;
        this->result();
        if (!this->passed)
            cout << "\tExpected: " << valueToBeExpected << "\n\tActual: " << this->outputString << endl;
        return *this;
    }

    Test &Expect(const char *value)
    {
        this->outputChar = value;
        this->passed = false;
        return *this;
    }

    Test &ToBe(const char *valueToBeExpected)
    {
        this->passed = this->outputChar == valueToBeExpected;
        this->result();
        if (!this->passed)
            cout << "\tExpected: " << valueToBeExpected << "\n\tActual: " << this->outputChar << endl;
        return *this;
    }

    Test &ExpectMatrix(int a1, int a2, int a3, int a4)
    {
        this->matrix[0] = a1;
        this->matrix[1] = a2;
        this->matrix[2] = a3;
        this->matrix[3] = a4;
        this->passed = false;
        return *this;
    }

    Test &ToBe(int a1, int a2, int a3, int a4)
    {

        this->passed = this->matrix[0] == a1 &&
                       this->matrix[1] == a2 &&
                       this->matrix[2] == a3 &&
                       this->matrix[3] == a4;
        this->result();
        if (!this->passed)
            cout << "\tExpected: [" << a1 << "," << a2 << "," << a3 << "," << a4 << "]\n\tActual: ["
                 << this->matrix[0] << "," << this->matrix[1] << "," << this->matrix[2] << "," << this->matrix[3] << "]" << endl;
        return *this;
    }

    Test &ExpectMatrix(int *a)
    {
        this->matrix[0] = a[0];
        this->matrix[1] = a[1];
        this->matrix[2] = a[2];
        this->matrix[3] = a[3];
        this->passed = false;
        return *this;
    }

    Test &ToBe(int *a)
    {

        this->passed = this->matrix[0] == a[0] &&
                       this->matrix[1] == a[1] &&
                       this->matrix[2] == a[2] &&
                       this->matrix[3] == a[3];
        this->result();
        if (!this->passed)
            cout << "\tExpected: [" << a[0] << "," << a[1] << "," << a[2] << "," << a[3] << "]\n\tActual: ["
                 << this->matrix[0] << "," << this->matrix[1] << "," << this->matrix[2] << "," << this->matrix[3] << "]" << endl;
        return *this;
    }
};
