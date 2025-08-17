#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sqlite_modern_cpp.h>
#include <catch2/catch.hpp>

using namespace  sqlite;
using namespace std;

TEST_CASE("shared connections work fine", "[shared_connection]") {
    database db(":memory:");

    {
        auto con = db.connection();

        {
            database db2(con);
            int test = 0;
            db2 << "select 1" >> test;
            REQUIRE(test == 1);
        }

        int test = 0;
        db << "select 1" >> test;
        REQUIRE(test == 1);
    }
}
