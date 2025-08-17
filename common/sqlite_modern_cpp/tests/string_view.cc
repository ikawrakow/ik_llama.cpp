#include <sqlite_modern_cpp.h>
#include <catch2/catch.hpp>


#ifdef MODERN_SQLITE_STRINGVIEW_SUPPORT
#include <string_view>

using namespace sqlite;
TEST_CASE("std::string_view works", "[string_view]") {
	database db(":memory:");
	db << "CREATE TABLE foo (a integer, b string);\n";
	const std::string_view test1 = "null terminated string view";
	db << "INSERT INTO foo VALUES (?, ?)" << 1 << test1;
	std::string str;
	db << "SELECT b from FOO where a=?;" << 1 >> str;
	REQUIRE(test1 == str);
	const char s[] = "hello world";
	std::string_view test2(&s[0], 2);
	db << "INSERT INTO foo VALUES (?,?)" << 2 << test2;
	db << "SELECT b from FOO where a=?" << 2 >> str;
	REQUIRE(str == "he");
}
#endif
