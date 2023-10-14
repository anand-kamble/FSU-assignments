#include <iostream>

class Complex {
private:
    double real;
    double imag;

public:
    Complex() : real(0), imag(0) {}

    Complex(double r, double i) : real(r), imag(i) {}

    // Overloading the + operator
    Complex operator + (const Complex& obj) const {
        Complex result;
        result.real = real + obj.real;
        result.imag = imag + obj.imag;
        return result;
    }

    // Overloading the << operator for easy printing
    friend std::ostream& operator << (std::ostream& out, const Complex& obj) {
        out << obj.real << " + " << obj.imag << "i";
        return out;
    }
};

int main() {
    Complex a(2.0, 3.0);
    Complex b(1.5, 2.5);

    // Using the overloaded + operator
    Complex c = a + b;

    // Using the overloaded << operator for printing
    std::cout << "Result: " << c << std::endl;

    return 0;
}
