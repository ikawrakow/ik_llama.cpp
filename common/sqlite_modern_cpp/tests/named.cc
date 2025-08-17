#include <iostream>
#include <cstdlib>
#include <sqlite_modern_cpp.h>
#include <catch2/catch.hpp>
using namespace sqlite;
using namespace std;


TEST_CASE("binding named parameters works", "[named]") {
    database db(":memory:");

    db << "CREATE TABLE foo (a,b);";

    int a = 1;
    db << "INSERT INTO foo VALUES (:first,:second)" << named_parameter(":second", 2) << named_parameter(":first", a);

    db << "SELECT b FROM foo WHERE a=?;" << 1 >> a;

    REQUIRE(a == 2);
}