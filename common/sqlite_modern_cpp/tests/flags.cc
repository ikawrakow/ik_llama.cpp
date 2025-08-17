#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sqlite_modern_cpp.h>
#include <sys/types.h>
#include <catch2/catch.hpp>
using namespace sqlite;
using namespace std;

struct TmpFile {
	string fname;

	TmpFile(): fname("./flags.db") { }
	~TmpFile() { remove(fname.c_str()); }
};
#ifdef _WIN32
#define OUR_UTF16 "UTF-16le"
#elif BYTE_ORDER == BIG_ENDIAN
#define OUR_UTF16 "UTF-16be"
#else
#define OUR_UTF16 "UTF-16le"
#endif

TEST_CASE("flags work", "[flags]") {
    TmpFile file;
    sqlite::sqlite_config cfg;
    std::string enc;
    SECTION("PRAGMA endcoding is UTF-8 for string literals") {
        database db(":memory:", cfg);
        db << "PRAGMA encoding;" >> enc;
        REQUIRE(enc == "UTF-8"); 
    }
    SECTION("encoding is UTF-16 for u"" prefixed string literals") {
        database db(u":memory:", cfg);
        db << "PRAGMA encoding;" >> enc;
        REQUIRE(enc == OUR_UTF16); 
    }
    SECTION("we can set encoding to UTF-8 with flags") {
        cfg.encoding = Encoding::UTF8;
        database db(u":memory:", cfg);
        db << "PRAGMA encoding;" >> enc;
        REQUIRE(enc == "UTF-8"); 
    }
    SECTION("we can set encoding to UTF-16 with flags") {
        cfg.encoding = Encoding::UTF16;
        database db(u":memory:", cfg);
        db << "PRAGMA encoding;" >> enc;
        REQUIRE(enc == OUR_UTF16); 
    }
    SECTION("we can set encoding to UTF-16 with flags for on disk databases") {
        cfg.encoding = Encoding::UTF16;
        database db(file.fname, cfg);
        db << "PRAGMA encoding;" >> enc;
        REQUIRE(enc == OUR_UTF16); 

    }
    SECTION("READONLY flag works") {
        {
            database db(file.fname, cfg);
            db << "CREATE TABLE foo (a string);";
            db << "INSERT INTO foo VALUES (?)" << "hello";
        }

        cfg.flags = sqlite::OpenFlags::READONLY;
        database db(file.fname, cfg);

        string str;
        db << "SELECT a FROM foo;" >> str;

        REQUIRE(str == "hello");

        bool failed = false;
        try {
            db << "INSERT INTO foo VALUES (?)" << "invalid";
        } catch(errors::readonly&) {
            failed = true;
        }
        REQUIRE(failed == true);
    }
}
