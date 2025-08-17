#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <sqlite_modern_cpp.h>
#include <catch2/catch.hpp>

using namespace sqlite;
using std::string;

struct TmpFile {
    string fname;

    TmpFile(): fname("./trycatchblocks.db") {}
    ~TmpFile() { remove(fname.c_str()); }
};


class DBInterface {
    database db;

public:
    DBInterface( const string& fileName ) : db( fileName ) { }

    void LogRequest( const string& username, const string& ip, const string& request ) 
    {
        try {
            auto timestamp = std::to_string( time( nullptr ) );

            db  <<
                "create table if not exists log_request ("
                "   _id integer primary key autoincrement not null,"
                "   username text,"
                "   timestamp text,"
                "   ip text,"
                "   request text"
                ");";
            db  << "INSERT INTO log_request (username, timestamp, ip, request) VALUES (?,?,?,?);" 
                << username
                << timestamp
                << ip
                << request;
        } catch ( const std::exception& e ) {
            std::cout << e.what() << std::endl;
        }
    }

    bool TestData( void ) {
        try {
            string username, timestamp, ip, request;

            db  << "select username, timestamp, ip, request from log_request where username = ?"
                << "test"
                >> std::tie(username, timestamp, ip, request);
                    
            if ( username == "test" && ip == "127.0.0.1" && request == "hello world" ) {
                return true;
            }
        } catch ( const std::exception& e ) {
            std::cout << e.what() << std::endl;
        }

        return false;
    }
};

TEST_CASE("try catch blocks", "[trycatchblocks]") {
    // --------------------------------------------------------------------------
    // -- Test if writing to disk works properly from within a catch block.
    // --------------------------------------------------------------------------
    try {
        throw "hello";
    }
    catch ( ... ) {
        TmpFile tmpF;
        DBInterface interf(tmpF.fname);
        interf.LogRequest( "test", "127.0.0.1", "hello world" );
        REQUIRE(interf.TestData() == true); 
    }
}
