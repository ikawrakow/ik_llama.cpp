#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <catch.hpp>

#include <sqlite_modern_cpp/sqlcipher.h>
using namespace sqlite;
using namespace std;

struct TmpFile
{
    string fname;

    TmpFile(): fname("./sqlcipher.db") { }
    ~TmpFile() { remove(fname.c_str()); }
};

TEST_CASE("sqlcipher works", "[sqlcipher]") {
    TmpFile file;
    sqlcipher_config config;
    {
        config.key = "DebugKey";
        sqlcipher_database db(file.fname, config);

        db << "CREATE TABLE foo (a integer, b string);";
        db << "INSERT INTO foo VALUES (?, ?)" << 1 << "hello";
        db << "INSERT INTO foo VALUES (?, ?)" << 2 << "world";

        string str;
        db << "SELECT b from FOO where a=?;" << 2 >> str;

        REQUIRE(str == "world");
    }

    bool failed = false;
    try {
        config.key = "DebugKey2";
        sqlcipher_database db(file.fname, config);
        db << "INSERT INTO foo VALUES (?, ?)" << 3 << "fail";
    } catch(const errors::notadb&) {
        failed = true;
        // Expected, wrong key
    }
    REQUIRE(failed == true);

    {
        config.key = "DebugKey";
        sqlcipher_database db(file.fname, config);
        db.rekey("DebugKey2");
    }
    {
        config.key = "DebugKey2";
        sqlcipher_database db(file.fname, config);
        db << "INSERT INTO foo VALUES (?, ?)" << 3 << "fail";
    }
}
