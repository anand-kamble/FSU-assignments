/*
 * Filename: table.cpp
 * Author: Chat-GPT
 * Date: 3rd December 2023
 * Description: Implementation of the Table class for creating and printing tables.
 *              Adapted from the original source at: [https://chat.openai.com/share/bbdf6ebf-5b48-45d0-bf1e-1082ee16aece]
 */
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

class Table {
public:
    Table(const std::vector<std::string>& headers) : headers(headers) {}

    void addRow(const std::vector<std::string>& row) {
        data.push_back(row);
    }

    void printTable() const {
        // Print top border
        printHorizontalBorder();

        // Print headers
        printRow(headers);

        // Print header and data separator
        printHorizontalBorder();

        // Print data
        for (const auto& row : data) {
            printRow(row);
        }

        // Print bottom border
        printHorizontalBorder();
    }

private:
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> data;

    void printRow(const std::vector<std::string>& row) const {
        // Print left border
        std::cout << "|";

        for (const auto& cell : row) {
            std::cout << std::setw(15) << cell << "|"; // Adjust the setw value based on your needs
        }

        // Print right border and move to the next line
        std::cout << std::endl;
    }

    void printHorizontalBorder() const {
        // Print the top or bottom border
        for (size_t i = 0; i < headers.size(); ++i) {
            std::cout << "+---------------";
        }

        // Print the right border and move to the next line
        std::cout << "+" << std::endl;
    }
};