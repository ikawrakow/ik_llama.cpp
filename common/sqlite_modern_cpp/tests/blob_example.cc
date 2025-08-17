#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <catch2/catch.hpp>
#include <sqlite_modern_cpp.h>
using namespace  sqlite;
using namespace std;

TEST_CASE("Blob does work", "[blob]") {
    database db(":memory:");

    db << "CREATE TABLE person (name TEXT, numbers BLOB);";
    db << "INSERT INTO person VALUES (?, ?)" << "bob" << vector<int> { 1, 2, 3, 4};
    db << "INSERT INTO person VALUES (?, ?)" << "jack" << vector<char> { '1', '2', '3', '4'};
    db << "INSERT INTO person VALUES (?, ?)" << "sara" << vector<double> { 1.0, 2.0, 3.0, 4.0};

    vector<int> numbers_bob;
    db << "SELECT numbers from person where name = ?;" << "bob" >> numbers_bob;

    REQUIRE(numbers_bob.size() == 4);
    REQUIRE((numbers_bob[0] == 1 && numbers_bob[1] == 2 && numbers_bob[2] == 3 && numbers_bob[3] == 4));

    vector<char> numbers_jack;
    db << "SELECT numbers from person where name = ?;" << "jack" >> numbers_jack;

    REQUIRE(numbers_jack.size() == 4);
    REQUIRE((numbers_jack[0] == '1' && numbers_jack[1] == '2' && numbers_jack[2] == '3' && numbers_jack[3] == '4'));


    vector<double> numbers_sara;
    db << "SELECT numbers from person where name = ?;" << "sara" >> numbers_sara;

    REQUIRE(numbers_sara.size() == 4);
    REQUIRE((numbers_sara[0] == 1.0 && numbers_sara[1] == 2.0 && numbers_sara[2] == 3.0 && numbers_sara[3] == 4.0));
}
