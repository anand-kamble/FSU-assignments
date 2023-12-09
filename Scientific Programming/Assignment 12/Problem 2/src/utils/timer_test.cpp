#include <array>
#include <chrono> // for std::chrono functions
#include <cstddef> // for std::size_t
#include <iostream>
#include <numeric> // for std::iota

const int g_arrayElements = 20000;

class Timer
{
private:
  // Type aliases to make accessing nested type easier
  using clock_type = std::chrono::steady_clock;
  using second_type = std::chrono::duration<double, std::ratio<1> >;

  std::chrono::time_point<clock_type> m_beg;

public:
  Timer() : m_beg ( clock_type::now() )
  {
  }

  void reset()
  {
    m_beg = clock_type::now();
  }

  double elapsed() const
  {
    return std::chrono::duration_cast<second_type>(clock_type::now() - m_beg).count();
  }
};
