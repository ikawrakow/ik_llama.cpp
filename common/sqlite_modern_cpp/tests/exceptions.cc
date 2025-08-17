#include <iostream>
#include <iomanip>
#include <string>
#include <memory>
#include <stdexcept>
#include <sqlite_modern_cpp.h>
#include <catch2/catch.hpp>
using namespace sqlite;
using namespace std;


TEST_CASE("exceptions are thrown", "[exceptions]") {
    database db(":memory:");
    db << "CREATE TABLE person (id integer primary key not null, name TEXT);";
    bool expception_thrown = false;
    std::string get_sql_result;

#if SQLITE_VERSION_NUMBER >= 3014000
    std::string expedted_sql =  "INSERT INTO person (id,name) VALUES (1,'jack')";
#else
    std::string expedted_sql =  "INSERT INTO person (id,name) VALUES (?,?)";
#endif

    SECTION("Parent exception works") {
        try {
            db << "INSERT INTO person (id,name) VALUES (?,?)" << 1 << "jack";
            // inserting again to produce error
            db << "INSERT INTO person (id,name) VALUES (?,?)" << 1 << "jack";
        } catch (errors::constraint& e) {
            expception_thrown = true;
            get_sql_result = e.get_sql();
        }

        REQUIRE(expception_thrown == true);
        REQUIRE(get_sql_result == expedted_sql);
    }

    SECTION("Extended exception works") {
        try {
            db << "INSERT INTO person (id,name) VALUES (?,?)" << 1 << "jack";
            // inserting again to produce error
            db << "INSERT INTO person (id,name) VALUES (?,?)" << 1 << "jack";
        } catch (errors::constraint_primarykey& e) {
            expception_thrown = true;
            get_sql_result = e.get_sql();
        }

        REQUIRE(expception_thrown == true);
        REQUIRE(get_sql_result == expedted_sql);
    }
}
