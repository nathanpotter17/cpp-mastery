#include <iostream>
#include <string>

int main () {
  int age;
  std::string name;
  std::cout << "Please type your name & age" << std::endl;
  std::getline(std::cin, name);    // std::cin >> age;
  std::cin >> age;
  std::cout << "Hello " << name << " you are " << age << " years old" << std::endl;
  return 0;
}
