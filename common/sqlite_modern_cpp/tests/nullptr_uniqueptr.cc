#include <iostream>
#include <string>
#include <vector>
#include <sqlite_modern_cpp.h>
#include <catch2/catch.hpp>
using namespace std;
using namespace sqlite;

TEST_CASE("nullptr & unique_ptr", "[null_ptr_unique_ptr]") {
    database db(":memory:");
    db << "CREATE TABLE tbl (id integer,age integer, name string, img blob);";
    db << "INSERT INTO tbl VALUES (?, ?, ?, ?);" << 1 << 24 << "bob" << vector<int> { 1, 2 , 3};
    unique_ptr<string> ptr_null;
    db << "INSERT INTO tbl VALUES (?, ?, ?, ?);" << 2 << nullptr << ptr_null << nullptr;

    db << "select age,name,img from tbl where id = 1" >> [](unique_ptr<int> age_p, unique_ptr<string> name_p, unique_ptr<vector<int>> img_p) {
        REQUIRE(age_p != nullptr);
        REQUIRE(name_p != nullptr);
        REQUIRE(img_p != nullptr);
    };

    db << "select age,name,img from tbl where id = 2" >> [](unique_ptr<int> age_p, unique_ptr<string> name_p, unique_ptr<vector<int>> img_p) {
        REQUIRE(age_p == nullptr);
        REQUIRE(name_p == nullptr);
        REQUIRE(img_p == nullptr);
    };
}
