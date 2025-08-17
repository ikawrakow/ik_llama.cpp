#include <iostream>
#include <iomanip>
#include <string>
#include <memory>
#include <stdexcept>
#include <sqlite_modern_cpp.h>
#include <sqlite_modern_cpp/log.h>
#include <catch2/catch.hpp>
using namespace sqlite;
using namespace std;

struct TrackErrors {
    TrackErrors() 
        : constraint_called{false}, primarykey_called{false}
    {
          error_log(
            [this](errors::constraint) {
                constraint_called = true;
            },
            [this](errors::constraint_primarykey e) {
                primarykey_called = true;
            }
            // We are not registering the unique key constraint:
            // For a unique key error the first handler (errors::constraint) will be called instead.
          );
    }

    bool constraint_called;
    bool primarykey_called;
    /* bool unique_called; */
};

// Run before main, before any other sqlite function.
static TrackErrors track;


TEST_CASE("error_log works", "[log]") {
  database db(":memory:");
  db << "CREATE TABLE person (id integer primary key not null, name TEXT unique);";

  SECTION("An extended error code gets called when registered") {
      try {
        db << "INSERT INTO person (id,name) VALUES (?,?)" << 1 << "jack";
        // triger primarykey constraint of 'id'
        db << "INSERT INTO person (id,name) VALUES (?,?)" << 1 << "bob";
      } catch (const errors::constraint& e) { }
      REQUIRE(track.primarykey_called == true);
      REQUIRE(track.constraint_called == false);
      track.primarykey_called = false;
  }

  SECTION("Parent gets called when the exact error code is not registered") {
      try {
        db << "INSERT INTO person (id,name) VALUES (?,?)" << 1 << "jack";
        // trigger unique constraint of 'name'
        db << "INSERT INTO person (id,name) VALUES (?,?)" << 2 << "jack";
      } catch (const errors::constraint& e) { }

      REQUIRE(track.primarykey_called == false);
      REQUIRE(track.constraint_called == true);
      track.constraint_called = false;
  }
}
