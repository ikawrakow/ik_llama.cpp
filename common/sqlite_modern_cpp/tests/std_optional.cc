#include <iostream>
#include <sqlite_modern_cpp.h>
#include <catch2/catch.hpp>

using namespace sqlite;
using namespace std;

#ifdef MODERN_SQLITE_STD_OPTIONAL_SUPPORT
TEST_CASE("std::optional works", "[optional]") {
    database db(":memory:");

    db << "drop table if exists test";
    db <<
        "create table if not exists test ("
        "   id integer primary key,"
        "   val int"
        ");";

    db << "insert into test(id,val) values(?,?)" << 1 << 5;
    db << "select id,val from test" >> [&](long long, sqlite::optional<int> val) {
        REQUIRE(val);
    };

    db << "delete from test where id = 1";
    db << "insert into test(id,val) values(?,?)" << 1 << nullptr;
    db << "select id,val from test" >> [&](long long, sqlite::optional<int> val) {
        REQUIRE(!val);
    };

}
#endif
