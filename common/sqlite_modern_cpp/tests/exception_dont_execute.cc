#include <iostream>
#include <string>
#include <memory>
#include <stdexcept>
#include <sqlite_modern_cpp.h>
#include <catch2/catch.hpp>
using namespace sqlite;
using namespace std;


TEST_CASE("Prepared statement will not execute on exceptions", "[prepared_statements]") {
	database db(":memory:");
	db << "CREATE TABLE person (id integer primary key not null, name TEXT not null);";

	try {
		auto stmt = db << "INSERT INTO person (id,name) VALUES (?,?)";
		throw 1;
	} catch (int) { }

    int count;
    db << "select count(*) from person" >> count;
    REQUIRE(count == 0);
}
