#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sqlite_modern_cpp.h>
#include <catch2/catch.hpp>
using namespace  sqlite;
using namespace std;

int add_integers(int i, int j) {
	return i+j;
}
TEST_CASE("sql functions work", "[functions]") {
    database db(":memory:");

    db.define("my_new_concat", [](std::string i, std::string j) {return i+j;});
    db.define("my_new_concat", [](std::string i, std::string j, std::string k) {return i+j+k;});
    db.define("add_integers", &add_integers);

    std::string test1, test3;
    int test2 = 0;
    db << "select my_new_concat('Hello ','world!')" >> test1;
    db << "select add_integers(1,1)" >> test2;
    db << "select my_new_concat('a','b','c')" >> test3;

    REQUIRE(test1 == "Hello world!");
    REQUIRE(test2 == 2);
    REQUIRE(test3 == "abc");

    db.define("my_count", [](int &i, int) {++i;}, [](int &i) {return i;});
    db.define("my_concat_aggregate", [](std::string &stored, std::string current) {stored += current;}, [](std::string &stored) {return stored;});

    db << "create table countable(i, s)";
    db << "insert into countable values(1, 'a')";
    db << "insert into countable values(2, 'b')";
    db << "insert into countable values(3, 'c')";
    db << "select my_count(i) from countable" >> test2;
    db << "select my_concat_aggregate(s) from countable order by i" >> test3;

    REQUIRE(test2 == 3);
    REQUIRE(test3 == "abc"); 

    db.define("tgamma", [](double i) {return std::tgamma(i);});
    db << "CREATE TABLE numbers (number INTEGER);";

    for(auto i=0; i!=10; ++i)
        db << "INSERT INTO numbers VALUES (?);" << i;

    db << "SELECT number, tgamma(number+1) FROM numbers;" >> [](double number, double factorial) {
        /* cout << number << "! = " << factorial << '\n'; */
    };
}
