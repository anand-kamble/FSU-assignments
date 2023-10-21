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

    bool result()
    {
        // cout << (this->passed ? "[Passed]" : "[Failed]") << " : " << this->name << endl;
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
};
