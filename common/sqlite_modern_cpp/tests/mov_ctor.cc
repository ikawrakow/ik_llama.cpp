// Fixing https://github.com/SqliteModernCpp/sqlite_modern_cpp/issues/63
#include <iostream>
#include <cstdlib>
#include <sqlite_modern_cpp.h>
#include <memory>
#include <catch2/catch.hpp>
using namespace  sqlite;
using namespace std;

struct dbFront {
	std::unique_ptr<database_binder> storedProcedure;
	database db;
	dbFront(): db(":memory:") {
	    db << "CREATE TABLE tbl (id integer, name string);";
	    // the temporary moved object should not run _execute() function on destruction.
		storedProcedure = std::make_unique<database_binder>(
			db << "INSERT INTO tbl VALUES (?, ?);"
		);
	}
};


TEST_CASE("database lifecycle", "move_ctor") {

  bool failed = false;
  try { dbFront dbf; }
  catch(const sqlite_exception& e) { failed = true; }
  catch(...) { failed = true; }

  REQUIRE(failed == false);
}
