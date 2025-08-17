#include<iostream>
#include<sqlite_modern_cpp.h>
#include<string>
#include<vector>
#include<catch2/catch.hpp>
using namespace  sqlite;
using namespace std;

template<typename Target, typename... AttrTypes>
struct builder {
    vector<Target> results;

    void operator()(AttrTypes... args) {
      results.emplace_back(std::forward<AttrTypes&&>(args)...);
    };
};


struct user {
	int age;
	string name;
	double weight;
	
	user(int age, string name, double weight) : age(age), name(name), weight(weight) { }

    static std::vector<user> all(sqlite::database& db) {
	  builder<user, int, std::string, double> person_builder;
      db << "SELECT * FROM user;"
        >> person_builder;
      return std::move(person_builder.results); // move to avoid copying data ;-)
    };
};

TEST_CASE("lvalue functors work", "[lvalue_functor]") {

    database db(":memory:");

    db <<
        "create table if not exists user ("
        "   age int,"
        "   name text,"
        "   weight real"
        ");";

    db << "insert into user (age,name,weight) values (?,?,?);" << 20 << u"chandler" << 83.25;
    db << "insert into user (age,name,weight) values (?,?,?);" << 21 << u"monika" << 86.25;
    db << "insert into user (age,name,weight) values (?,?,?);" << 22 << u"ross" << 88.25;

    auto users = user::all(db);

    REQUIRE(users.size() == 3);
}
