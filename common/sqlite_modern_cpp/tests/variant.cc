#include <iostream>
#include <cstdlib>
#include <sqlite_modern_cpp.h>
#include <catch2/catch.hpp>
using namespace sqlite;
using namespace std;

#ifdef MODERN_SQLITE_STD_VARIANT_SUPPORT
TEST_CASE("std::variant works", "[variant]") {
    database db(":memory:");

    db << "CREATE TABLE foo (a);";
    std::variant<std::string, int, std::optional<float>> v;

    v = 1;
    db << "INSERT INTO foo VALUES (?)" << v;

    v = "a";
    db << "INSERT INTO foo VALUES (?)" << v;

    db << "SELECT a FROM foo WHERE a=?;" << 1 >> v;

    REQUIRE(v.index() == 1);
    REQUIRE(std::get<1>(v) == 1);

    db << "SELECT NULL" >> v;
    REQUIRE(!std::get<2>(v)); 

    db << "SELECT 0.0" >> v;
    REQUIRE(std::get<2>(v)); 
}

TEST_CASE("std::monostate is a nullptr substitute", "[monostate]") {
    database db(":memory:");
    db << "CREATE TABLE foo (a);";

    std::variant<std::monostate,std::string> v;
    v=std::monostate();
    db << "INSERT INTO foo VALUES (?)" << v;
    db << "INSERT INTO foo VALUES (?)" << "This isn't a monostate!";

    bool found_null   = false,
         found_string = false;

    db << "SELECT * FROM foo"
    	 >> [&](std::variant<std::monostate, std::string> z) {
      if(z.index() == 0) {
        found_null = true;
      } else {
        found_string = true;
      }
    };
    REQUIRE((found_null && found_string));
    db << "SELECT NULL" >> v;
    REQUIRE(v.index() == 0);
}
#endif
