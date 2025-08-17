#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sqlite_modern_cpp.h>
#include <catch2/catch.hpp>

using namespace sqlite;
using namespace std;

struct tbl_functor {
  explicit tbl_functor(vector<pair<int, string> > &vec_) : vec(vec_) { }

  void operator() ( int id, string name) {
    vec.push_back(make_pair(id, move(name)));
  }
  vector<pair<int,string> > &vec;
};

TEST_CASE("functors work", "[functors]") {
    database db(":memory:");
    db << "CREATE TABLE tbl (id integer, name string);";
    db << "INSERT INTO tbl VALUES (?, ?);" << 1 << "hello";
    db << "INSERT INTO tbl VALUES (?, ?);" << 2 << "world";

    vector<pair<int,string> > vec;
    db << "select id,name from tbl;" >> tbl_functor(vec);

    REQUIRE(vec.size() == 2);

    vec.clear();

    tbl_functor functor(vec);
    db << "select id,name from tbl;" >> functor;

    REQUIRE(vec.size() == 2);
    REQUIRE(vec[0].first == 1);
    REQUIRE(vec[0].second == "hello");
}
