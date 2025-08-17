#define CATCH_CONFIG_MAIN
#include<iostream>
#include <catch2/catch.hpp>
#include <sqlite_modern_cpp.h>

using namespace sqlite;
using namespace std;

TEST_CASE("README Example Works", "[readme]") {

    database db(":memory:");

    db <<
        "create table if not exists user ("
        "   _id integer primary key autoincrement not null,"
        "   age int,"
        "   name text,"
        "   weight real"
        ");";

    db << "insert into user (age,name,weight) values (?,?,?);"
        << 20
        << u"bob"
        << 83.25;

    int age = 22; float weight = 68.5; string name = "jack";
    db << u"insert into user (age,name,weight) values (?,?,?);" // utf16 query string
        << age
        << name
        << weight;

    REQUIRE(db.last_insert_rowid() != 0);

    db << "select age,name,weight from user where age > ? ;"
        << 21
        >> [&](int _age, string _name, double _weight) {
            REQUIRE((_age == age && _name == name));
    };

    for(auto &&row : db << "select age,name,weight from user where age > ? ;" << 21) {
      int _age;
      string _name;
      double _weight;
      row >> _age >> _name >> _weight;
      REQUIRE((_age == age && _name == name));
    }

    for(std::tuple<int, string, double> row : db << "select age,name,weight from user where age > ? ;" << 21) {
      REQUIRE((std::get<int>(row) == age && std::get<string>(row) == name));
    }

    // selects the count(*) from user table
    // note that you can extract a single culumn single row result only to : int,long,long,float,double,string,u16string
    int count = 0;
    db << "select count(*) from user" >> count;
    REQUIRE(count == 2);

    db << "select age, name from user where _id=1;" >> tie(age, name);

    // this also works and the returned value will be automatically converted to string
    string str_count;
    db << "select count(*) from user" >> str_count;
    REQUIRE(str_count == string{"2"});
}
