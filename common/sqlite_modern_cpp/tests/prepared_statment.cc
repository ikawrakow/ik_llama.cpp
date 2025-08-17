#include <iostream>
#include <cstdlib>
#include <sqlite_modern_cpp.h>
#include <catch2/catch.hpp>
using namespace  sqlite;
using namespace std;

TEST_CASE("prepared statements work", "[prepared_statement]") {
    database db(":memory:");

    auto pps = db << "select ?"; // get a prepared parsed and ready statment

    int test = 4;
    pps << test; // set a bound var

    pps >> test; // execute statement

    REQUIRE(test == 4);

    pps << 4; // bind a rvalue
    pps++; // and execute

    pps << 8 >> test;

    REQUIRE(test == 8);

    auto pps2 = db << "select 1,2"; // multiple extract test

    pps2 >> [](int a, int b) {
        REQUIRE(a == 1);
        REQUIRE(b == 2);
    };

    auto pps3 = db << "select ?,?,?";

    test = 2;
    pps3 << 1 << test << 3 >> [](int a, int b, int c) {
        REQUIRE(a == 1);
        REQUIRE(b == 2);
        REQUIRE(c == 3);
    };

    test = 1;
    db << "select ?,?" << test << 5 >> test; // and mow everything together
    REQUIRE(test == 1);

    test = 2;
    db << "select ?,?,?" << 1 << test << 3 >> [](int a, int b, int c) {
        REQUIRE(a == 1);
        REQUIRE(b == 2);
        REQUIRE(c == 3);
    };

    db << "select ?" << test; 		// noVal		
    db << "select ?,?" << test << 1;
    db << "select ?,?" << 1 << test;
    db << "select ?,?" << 1 << 1;
    db << "select ?,?" << test << test;

    db << "select ?" << test >> test; 		// lVal		
    db << "select ?,?" << test << 1 >> test;
    db << "select ?,?" << 1 << test >> test;
    db << "select ?,?" << 1 << 1 >> test;
    db << "select ?,?" << test << test >> test;

    int q = 0;
    test = 1;
    db << "select ?" << test >> [&](int t) { q = t; }; 		// rVal		
    REQUIRE(q == 1);

    db << "select ?,?" << test << 1 >> [&](int t, int p) { q = t + p; };
    db << "select ?,?" << 1 << test >> [&](int t, int p) { q = t + p; };
    db << "select ?,?" << 1 << 1 >> [&](int t, int p) { q = t + p; };
    db << "select ?,?" << test << test >> [&](int t, int p) { q = t + p; };

    db << "select ?,?,?" << test << 1 << test; // mix
    db << "select ?,?,?" << 1 << test << 1;
    db << "select ?,?,?" << 1 << 1 << test;
    db << "select ?,?,?" << 1 << 1 << 1;
    db << "select ?,?,?" << test << test << test;

    {
        auto pps4 = db << "select ?,?,?"; // reuse

        (pps4 << test << 1 << test)++;
        (pps4 << 1 << test << 1)++;
        (pps4 << 1 << 1 << test)++;
        (pps4 << 1 << 1 << 1)++;
        (pps4 << test << test << test)++;
    }

    {
        auto prep = db << "select ?";

        prep << 5;
        prep.execute();
        prep << 6;
        prep.execute();
    }


}
